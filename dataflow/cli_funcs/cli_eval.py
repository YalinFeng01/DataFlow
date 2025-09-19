# dataflow/cli_funcs/cli_eval.py

import os
import shutil
import argparse
import subprocess
import sys
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataflow.cli_funcs.paths import DataFlowPath
from dataflow import get_logger
from dataflow.serving import LocalModelLLMServing_vllm, APILLMServing_request
from dataflow.operators.core_text import BenchDatasetEvaluator
from dataflow.utils.storage import FileStorage

logger = get_logger()


# =============================================================================
# 评估管道 - 集中的业务逻辑
# =============================================================================

class EvaluationPipeline:
    """评估管道：从模型检测到评估报告的完整流程"""
    
    def __init__(self, config: Dict[str, Any], cli_args=None):
        self.config = config
        self.cli_args = cli_args or argparse.Namespace()
        self.detected_models = []
        self.prepared_models = []
        self.generated_files = []
        
    def run(self) -> bool:
        """执行完整的评估流程"""
        try:
            # 1. 模型检测和准备
            self.detected_models = self._detect_models()
            if not self.detected_models:
                self._show_no_models_help()
                return False
                
            self.prepared_models = self._prepare_models()
            
            # 2. 生成答案
            self.generated_files = self._generate_answers()
            
            # 3. 执行评估
            results = self._run_evaluation()
            
            # 4. 生成报告
            self._generate_report(results)
            
            logger.info("✅ 评估流程完成")
            return True
            
        except Exception as e:
            logger.error(f"❌ 评估失败：{str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def _detect_models(self) -> List[Dict[str, Any]]:
        """检测可用的模型 - 重新设计为base vs fine-tuned对比"""
        models = []
        
        # 检查命令行参数是否指定了模型
        if hasattr(self.cli_args, 'models') and self.cli_args.models:
            logger.info("使用命令行指定的模型...")
            model_list = self.cli_args.models.split(',')
            for model_spec in model_list:
                model_spec = model_spec.strip()
                models.append({
                    "name": Path(model_spec).name,
                    "path": model_spec,
                    "type": "manual",
                    "needs_merge": False
                })
        elif hasattr(self.cli_args, 'no_auto') and self.cli_args.no_auto:
            # 禁用自动检测，使用配置文件中的手动模型
            logger.info("使用配置文件中手动指定的模型...")
            manual_models = self.config.get("TARGET_MODELS", {}).get("models", [])
            for model_spec in manual_models:
                models.append({
                    "name": Path(model_spec).name,
                    "path": model_spec,
                    "type": "manual",
                    "needs_merge": False
                })
        elif self.config.get("TARGET_MODELS", {}).get("auto_detect", True):
            # 自动检测模式 - 新逻辑
            logger.info("自动检测模式：寻找base model和fine-tuned model对比...")
            
            fine_tuned_models = []
            
            # 1. 检测微调模型
            saves_dir = Path("./.cache/saves")
            if saves_dir.exists():
                for model_dir in saves_dir.iterdir():
                    if model_dir.is_dir():
                        model_info = self._analyze_model_dir(model_dir)
                        if model_info:
                            fine_tuned_models.append(model_info)
            
            if fine_tuned_models:
                # 2. 尝试找到对应的base model
                base_model = self._find_base_model_for_finetuned(fine_tuned_models[0])
                
                if base_model:
                    # 找到了配对，进行有意义的对比
                    models.append(base_model)
                    models.append(fine_tuned_models[0])
                    logger.info(f"检测到对比对：")
                    logger.info(f"  Base model: {base_model['name']}")
                    logger.info(f"  Fine-tuned: {fine_tuned_models[0]['name']}")
                else:
                    # 没找到base model，给出提示
                    logger.warning("检测到微调模型，但未找到对应的base model")
                    self._show_base_model_missing_help(fine_tuned_models[0])
                    return []
            else:
                # 没有检测到任何微调模型
                logger.info("未检测到微调模型")
                self._show_no_models_help()
                return []
        else:
            # 使用配置文件中的手动模型
            manual_models = self.config.get("TARGET_MODELS", {}).get("models", [])
            for model_spec in manual_models:
                models.append({
                    "name": Path(model_spec).name,
                    "path": model_spec,
                    "type": "manual", 
                    "needs_merge": False
                })
        
        # 显示最终结果
        if models:
            logger.info(f"将评估以下模型对比:")
            for i, model in enumerate(models, 1):
                model_type = " (Base)" if i == 1 and len(models) == 2 else " (Fine-tuned)" if i == 2 and len(models) == 2 else ""
                merge_status = " (需要merge)" if model.get("needs_merge") else ""
                logger.info(f"  {i}. {model['name']}{model_type}{merge_status}")
        
        return models
    
    def _find_base_model_for_finetuned(self, fine_tuned_model: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """为微调模型寻找对应的base model"""
        
        # 如果是LoRA适配器，从adapter_config.json中读取base model
        if fine_tuned_model.get("type") == "lora_adapter":
            adapter_config_file = Path(fine_tuned_model["path"]) / "adapter_config.json"
            if adapter_config_file.exists():
                try:
                    with open(adapter_config_file, 'r') as f:
                        adapter_config = json.load(f)
                    base_model_path = adapter_config.get("base_model_name_or_path")
                    
                    if base_model_path:
                        return {
                            "name": Path(base_model_path).name,
                            "path": base_model_path,
                            "type": "base_model",
                            "needs_merge": False
                        }
                except Exception as e:
                    logger.warning(f"读取adapter配置失败: {e}")
        
        # TODO: 可以添加其他启发式方法来推断base model
        # 比如根据模型名称模式、训练日志等
        
        return None
    
    def _show_base_model_missing_help(self, fine_tuned_model: Dict[str, Any]):
        """显示缺少base model时的帮助信息"""
        print(f"检测到微调模型: {fine_tuned_model['name']}")
        print("但未找到对应的base model进行有意义的对比评估")
        print()
        print("请在 eval_api.py 配置文件中手动指定要对比的模型:")
        print("TARGET_MODELS = {")
        print("    'auto_detect': False,")
        print("    'models': [")
        print("        'Qwen/Qwen2.5-7B-Instruct',  # base model")
        print(f"        '{fine_tuned_model['path']}',  # 你的微调模型")
        print("    ]")
        print("}")
        print()
        print("这样能评估微调前后的性能提升效果")
    
    def _analyze_model_dir(self, model_dir: Path) -> Optional[Dict[str, Any]]:
        """分析模型目录，判断模型类型"""
        if not model_dir.exists():
            return None
            
        # 检查LoRA适配器文件
        adapter_files = ["adapter_config.json", "adapter_model.bin", "adapter_model.safetensors"]
        has_adapter = any((model_dir / f).exists() for f in adapter_files)
        
        # 检查基础模型文件
        model_files = ["config.json", "pytorch_model.bin", "model.safetensors"]
        has_model = any((model_dir / f).exists() for f in model_files)
        
        if has_adapter:
            return {
                "name": model_dir.name,
                "path": str(model_dir),
                "type": "lora_adapter",
                "needs_merge": True
            }
        elif has_model:
            return {
                "name": model_dir.name, 
                "path": str(model_dir),
                "type": "full_model",
                "needs_merge": False
            }
        
        return None
    
    def _prepare_models(self) -> List[Dict[str, Any]]:
        """准备模型：处理merge等预处理步骤"""
        prepared = []
        
        for model in self.detected_models:
            try:
                if model.get("needs_merge"):
                    logger.info(f"正在merge模型：{model['name']}")
                    merged_path = self._merge_lora_model(model["path"])
                    # 更新模型路径和名称为实际的merged路径
                    model["path"] = merged_path
                    model["name"] = Path(merged_path).name  # 使用实际的merged目录名
                    model["needs_merge"] = False
                    logger.info(f"Merge完成：{merged_path}")
                
                prepared.append(model)
                
            except Exception as e:
                logger.error(f"模型 {model['name']} 准备失败，将跳过：{e}")
                continue
        
        return prepared
    
    def _merge_lora_model(self, adapter_path: str) -> str:
        """合并LoRA适配器"""
        try:
            adapter_dir = Path(adapter_path)
            merged_dir = adapter_dir.parent / f"{adapter_dir.name}_merged"
            
            # 如果已经merge过了，直接返回
            if merged_dir.exists() and (merged_dir / "config.json").exists():
                logger.info(f"发现已存在的merged模型: {merged_dir}")
                return str(merged_dir)
            
            # 读取adapter配置
            adapter_config_file = adapter_dir / "adapter_config.json"
            if not adapter_config_file.exists():
                raise Exception(f"找不到adapter_config.json: {adapter_config_file}")
                
            with open(adapter_config_file, 'r') as f:
                adapter_config = json.load(f)
            base_model_path = adapter_config.get("base_model_name_or_path")
            
            if not base_model_path:
                raise Exception("adapter_config.json中缺少base_model_name_or_path")
                
            logger.info(f"Base model: {base_model_path}")
            logger.info(f"开始merge LoRA适配器...")
            
            # 尝试多种merge方法
            merge_success = False
            
            # 方法1: 使用llamafactory export
            try:
                cmd = [
                    "llamafactory-cli", "export",
                    "--model_name_or_path", base_model_path,
                    "--adapter_name_or_path", str(adapter_dir),
                    "--template", "default",
                    "--finetuning_type", "lora", 
                    "--export_dir", str(merged_dir),
                    "--export_size", "2",
                    "--export_device", "cpu"
                ]
                
                logger.info("尝试使用llamafactory export进行merge...")
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                
                if merged_dir.exists() and (merged_dir / "config.json").exists():
                    logger.info(f"llamafactory merge成功: {merged_dir}")
                    merge_success = True
                
            except (subprocess.CalledProcessError, FileNotFoundError) as e:
                logger.warning(f"llamafactory export失败: {e}")
                if merged_dir.exists():
                    import shutil
                    shutil.rmtree(merged_dir)
            
            # 方法2: 使用transformers的PeftModel
            if not merge_success:
                try:
                    logger.info("尝试使用transformers PeftModel进行merge...")
                    
                    # 这需要在运行时导入，避免依赖问题
                    from transformers import AutoModelForCausalLM, AutoTokenizer
                    from peft import PeftModel
                    import torch
                    
                    # 加载base model
                    logger.info(f"加载base model: {base_model_path}")
                    base_model = AutoModelForCausalLM.from_pretrained(
                        base_model_path,
                        torch_dtype=torch.float16,
                        device_map="auto",
                        trust_remote_code=True
                    )
                    
                    # 加载并merge LoRA
                    logger.info(f"加载LoRA适配器: {adapter_dir}")
                    model = PeftModel.from_pretrained(base_model, str(adapter_dir))
                    model = model.merge_and_unload()
                    
                    # 保存merged模型
                    logger.info(f"保存merged模型到: {merged_dir}")
                    merged_dir.mkdir(exist_ok=True)
                    model.save_pretrained(str(merged_dir))
                    
                    # 复制tokenizer
                    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
                    tokenizer.save_pretrained(str(merged_dir))
                    
                    if (merged_dir / "config.json").exists():
                        logger.info("transformers merge成功")
                        merge_success = True
                    
                    # 清理内存
                    del model, base_model, tokenizer
                    torch.cuda.empty_cache()
                    
                except Exception as e:
                    logger.warning(f"transformers merge失败: {e}")
                    if merged_dir.exists():
                        import shutil
                        shutil.rmtree(merged_dir)
            
            if merge_success:
                return str(merged_dir)
            else:
                raise Exception("所有merge方法都失败了")
                
        except Exception as e:
            logger.error(f"LoRA merge失败: {e}")
            logger.info("将跳过该模型的评估")
            raise e
    
    def _generate_answers(self) -> List[Dict[str, str]]:
        """为所有模型生成答案"""
        data_config = self.config.get("DATA_CONFIG", {})
        input_file = data_config.get("input_file", "./.cache/data/qa.json")
        
        if not Path(input_file).exists():
            raise FileNotFoundError(f"数据文件不存在：{input_file}")
        
        # 读取问题数据
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        question_key = data_config.get("question_key", "instruction")
        questions = [item[question_key] for item in data]
        
        generated_files = []
        batch_size = self.config.get("EVAL_CONFIG", {}).get("batch_size", 8)
        max_tokens = self.config.get("EVAL_CONFIG", {}).get("max_tokens", 512)
        
        for model in self.prepared_models:
            logger.info(f"🤖 为模型 {model['name']} 生成答案...")
            
            try:
                # 加载模型 - 简化版本，只处理已经prepare好的模型
                model_serving = LocalModelLLMServing_vllm(
                    hf_model_name_or_path=model["path"],
                    vllm_tensor_parallel_size=1,
                    vllm_max_tokens=max_tokens
                )
                
                # 批量生成答案
                answers = []
                for i in range(0, len(questions), batch_size):
                    batch_questions = questions[i:i+batch_size]
                    batch_answers = model_serving.generate_from_input(batch_questions)
                    answers.extend(batch_answers)
                    logger.info(f"  进度: {min(i+batch_size, len(questions))}/{len(questions)}")
                
                # 保存带答案的数据
                output_data = data.copy()
                for i, item in enumerate(output_data):
                    item["model_generated_answer"] = answers[i]
                
                # 生成安全的文件名
                safe_name = "".join(c for c in model['name'] if c.isalnum() or c in ('-', '_'))
                output_file = f"qa_{safe_name}_answers.json"
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(output_data, f, ensure_ascii=False, indent=2)
                
                generated_files.append({
                    "model_name": model['name'],
                    "file_path": output_file
                })
                
                logger.info(f"✅ 答案已保存：{output_file}")
                
            except Exception as e:
                logger.error(f"❌ 模型 {model['name']} 答案生成失败：{e}")
                continue
        
        return generated_files
    
    def _run_evaluation(self) -> List[Dict[str, Any]]:
        """运行评估"""
        data_config = self.config.get("DATA_CONFIG", {})
        eval_config = self.config.get("EVAL_CONFIG", {})
        
        # 创建评估器LLM服务
        logger.info("创建评估器LLM服务...")
        try:
            judge_serving = self.config["create_judge_serving"]()
            logger.info("评估器服务创建成功")
        except Exception as e:
            logger.error(f"评估器服务创建失败: {e}")
            return []
        
        results = []
        
        for file_info in self.generated_files:
            logger.info(f"📊 评估模型：{file_info['model_name']}")
            
            try:
                # 创建存储
                cache_name = "".join(c for c in file_info['model_name'] if c.isalnum() or c in ('-', '_'))
                storage = self.config["create_storage"](
                    file_info["file_path"], 
                    f"./eval_cache_{cache_name}"
                )
                
                # 创建评估器
                result_file = f"./eval_result_{cache_name}.json"
                evaluator = self.config["create_evaluator"](judge_serving, result_file)
                
                logger.info(f"开始评估，数据源: {file_info['file_path']}")
                logger.info(f"评估方法: {eval_config.get('compare_method', 'semantic')}")
                logger.info(f"结果将保存到: {result_file}")
                
                # 运行评估
                try:
                    evaluator.run(
                        storage=storage.step(),
                        input_test_answer_key="model_generated_answer",
                        input_gt_answer_key=data_config.get("reference_answer_key", "output"),
                        input_question_key=data_config.get("question_key", "instruction")
                    )
                    logger.info(f"模型 {file_info['model_name']} 评估完成")
                except Exception as eval_error:
                    logger.error(f"评估执行失败: {eval_error}")
                    continue
                
                # 读取结果
                if Path(result_file).exists():
                    with open(result_file, 'r', encoding='utf-8') as f:
                        result_data = json.load(f)
                        if isinstance(result_data, list) and result_data:
                            result_data[0]["model_name"] = file_info['model_name']
                            results.append(result_data[0])
                            
            except Exception as e:
                logger.error(f"❌ 模型 {file_info['model_name']} 评估失败：{e}")
                continue
        
        return results
    
    def _generate_report(self, results: List[Dict[str, Any]]):
        """生成对比报告 - 显示真实的模型名称"""
        if not results:
            logger.warning("没有有效的评估结果")
            return
            
        logger.info("生成评估报告...")
        
        # 控制台输出
        print("\n" + "="*60)
        print("模型评估结果对比")
        print("="*60)
        
        # 按准确率排序
        sorted_results = sorted(results, key=lambda x: x.get("accuracy", 0), reverse=True)
        
        # 智能识别base model和fine-tuned model
        base_result = None
        finetuned_result = None
        
        if len(sorted_results) == 2:
            # 尝试识别哪个是base model，哪个是fine-tuned model
            for result in sorted_results:
                model_name = result.get('model_name', '').lower()
                if any(keyword in model_name for keyword in ['qwen', 'llama', 'chatglm', 'baichuan']) and 'cache' not in model_name:
                    base_result = result
                elif 'cache' in model_name or 'merged' in model_name:
                    finetuned_result = result
        
        if base_result and finetuned_result:
            # 有意义的base vs fine-tuned对比
            print("Base Model vs Fine-tuned Model 对比:")
            print(f"📊 Base Model: {base_result.get('model_name', 'Unknown')}")
            print(f"   准确率: {base_result.get('accuracy', 0):.3f}")
            print(f"   样本数: {base_result.get('total_samples', 0)}")
            print(f"   匹配数: {base_result.get('matched_samples', 0)}")
            print()
            print(f"🚀 Fine-tuned Model: {finetuned_result.get('model_name', 'Unknown')}")
            print(f"   准确率: {finetuned_result.get('accuracy', 0):.3f}")
            print(f"   样本数: {finetuned_result.get('total_samples', 0)}")
            print(f"   匹配数: {finetuned_result.get('matched_samples', 0)}")
            print()
            
            # 计算提升
            base_acc = base_result.get('accuracy', 0)
            ft_acc = finetuned_result.get('accuracy', 0)
            improvement = ft_acc - base_acc
            improvement_pct = (improvement / base_acc * 100) if base_acc > 0 else 0
            
            if improvement > 0:
                print(f"✅ 微调效果: +{improvement:.3f} ({improvement_pct:+.1f}%)")
                print("🎉 微调成功提升了模型性能!")
            elif improvement < 0:
                print(f"❌ 微调效果: {improvement:.3f} ({improvement_pct:+.1f}%)")
                print("⚠️  微调可能存在过拟合或数据质量问题")
            else:
                print("➡️  微调效果: 无显著变化")
        else:
            # 通用的多模型对比
            print("多模型性能对比:")
            for i, result in enumerate(sorted_results, 1):
                print(f"{i}. {result.get('model_name', 'Unknown')}")
                print(f"   准确率: {result.get('accuracy', 0):.3f}")
                print(f"   样本数: {result.get('total_samples', 0)}")
                print(f"   匹配数: {result.get('matched_samples', 0)}")
                print()
        
        # 保存详细报告
        judge_model_name = "unknown"
        judge_config = self.config.get("JUDGE_MODEL_CONFIG", {})
        if "model_name" in judge_config:
            judge_model_name = judge_config["model_name"]
        elif "model_path" in judge_config:
            judge_model_name = Path(judge_config["model_path"]).name
        
        report = {
            "evaluation_summary": {
                "judge_model": judge_model_name,
                "evaluation_type": "base_vs_finetuned" if (base_result and finetuned_result) else "multi_model",
                "total_models": len(results),
                "results": sorted_results
            }
        }
        
        # 如果是base vs fine-tuned对比，添加额外信息
        if base_result and finetuned_result:
            report["evaluation_summary"]["comparison"] = {
                "base_model": {
                    "name": base_result.get('model_name'),
                    "accuracy": base_result.get('accuracy', 0)
                },
                "finetuned_model": {
                    "name": finetuned_result.get('model_name'),
                    "accuracy": finetuned_result.get('accuracy', 0)
                },
                "improvement": {
                    "absolute": improvement,
                    "percentage": improvement_pct,
                    "status": "improved" if improvement > 0 else "degraded" if improvement < 0 else "unchanged"
                }
            }
        
        report_file = "model_comparison_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"详细报告已保存：{report_file}")
        print("="*60)
    
    def _show_no_models_help(self):
        """显示无模型时的帮助信息"""
        print("❌ 未检测到可用的模型进行评估")
        print()
        print("解决方案:")
        print("1. 训练模型:")
        print("   dataflow text2model init && dataflow text2model train")
        print()
        print("2. 手动指定模型 - 编辑配置文件:")
        print("   TARGET_MODELS = {")
        print("       'auto_detect': False,")
        print("       'models': [")
        print("           'Qwen/Qwen2.5-7B-Instruct',")
        print("           'meta-llama/Llama-3-8B-Instruct'")
        print("       ]")
        print("   }")
        print()
        print("3. 使用命令行参数指定模型:")
        print("   dataflow eval api --models Qwen/Qwen2.5-7B-Instruct,meta-llama/Llama-3-8B-Instruct")


class DataFlowEvalCLI:
    """DataFlow 评估命令行工具 - 复制算子配置文件模式"""
    
    def __init__(self):
        self.eval_template_dir = Path(DataFlowPath.get_dataflow_dir()) / "cli_funcs" / "eval_pipeline"
        self.current_dir = Path.cwd()
    
    def init_eval_file(self, eval_type: str = "api", output_file: str = None):
        """复制评估算子配置文件到当前目录
        
        Args:
            eval_type: 评估类型 ("api" 或 "local")
            output_file: 输出文件名，默认为 eval_api.py 或 eval_local.py
        """
        if eval_type not in ["api", "local"]:
            logger.error("评估类型必须是 'api' 或 'local'")
            return False
        
        # 确定模板文件和输出文件
        template_file = self.eval_template_dir / f"eval_{eval_type}.py"
        if output_file is None:
            output_file = f"eval_{eval_type}.py"
        
        output_path = self.current_dir / output_file
        
        # 检查模板文件是否存在
        if not template_file.exists():
            logger.error(f"模板文件不存在：{template_file}")
            return False
        
        # 检查输出文件是否已存在
        if output_path.exists():
            logger.warning(f"文件 {output_file} 已存在，是否覆盖？(y/n)")
            user_input = input().strip().lower()
            if user_input != 'y':
                logger.info("操作取消")
                return False
        
        try:
            # 复制文件
            shutil.copy2(template_file, output_path)
            logger.info(f"✅ 评估配置文件已复制到：{output_path}")
            logger.info(f"💡 请编辑 {output_file} 文件配置您的评估参数，然后运行：")
            logger.info(f"   dataflow eval {eval_type} {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"复制文件失败：{str(e)}")
            return False
    
    def run_eval_file(self, eval_type: str, eval_file: str, cli_args=None):
        """运行评估文件
        
        Args:
            eval_type: 评估类型 ("api" 或 "local")
            eval_file: 评估文件路径
            cli_args: 命令行参数
        """
        eval_path = Path(eval_file)
        
        # 检查文件是否存在
        if not eval_path.exists():
            logger.error(f"评估文件不存在：{eval_file}")
            logger.info(f"请先运行 'dataflow eval init --type {eval_type}' 生成配置文件")
            return False
        
        # 检查文件是否为Python文件
        if eval_path.suffix != '.py':
            logger.error(f"评估文件必须是 Python 文件：{eval_file}")
            return False
        
        try:
            logger.info(f"开始运行 {eval_type} 模型评估：{eval_file}")
            
            # 动态导入用户的配置文件
            import sys
            import importlib.util
            
            # 获取文件的绝对路径
            config_path = eval_path.resolve()
            
            # 动态导入模块
            spec = importlib.util.spec_from_file_location("user_eval_config", config_path)
            if spec is None or spec.loader is None:
                logger.error(f"无法加载配置文件：{eval_file}")
                return False
                
            user_config_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(user_config_module)
            
            # 获取用户配置
            if hasattr(user_config_module, 'get_evaluator_config'):
                config = user_config_module.get_evaluator_config()
                
                # 创建并运行评估管道
                pipeline = EvaluationPipeline(config, cli_args)
                return pipeline.run()
                    
            else:
                logger.error(f"配置文件 {eval_file} 中没有找到 get_evaluator_config 函数")
                return False
                
        except Exception as e:
            logger.error(f"运行评估文件失败：{str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def list_eval_files(self):
        """列出当前目录下的评估文件"""
        eval_files = list(self.current_dir.glob("eval_*.py"))
        if eval_files:
            logger.info("找到以下评估配置文件：")
            for eval_file in eval_files:
                logger.info(f"  - {eval_file.name}")
        else:
            logger.info("当前目录下没有找到评估配置文件")
            logger.info("请运行 'dataflow eval init' 生成配置文件")


# =============================================================================
# 评估执行逻辑 - 保持原有接口兼容性
# =============================================================================

def run_api_evaluation(config):
    """运行API模型评估"""
    logger.info("DataFlow API模型评估开始")
    
    try:
        pipeline = EvaluationPipeline(config)
        success = pipeline.run()
        
        if success:
            logger.info("✅ API模型评估完成")
        else:
            logger.error("❌ API模型评估失败")
            
        return success
        
    except Exception as e:
        logger.error(f"❌ 评估失败：{str(e)}")
        raise

def run_local_evaluation(config):
    """运行本地模型评估"""
    logger.info("DataFlow 本地模型评估开始")
    
    try:
        pipeline = EvaluationPipeline(config)
        success = pipeline.run()
        
        if success:
            logger.info("✅ 本地模型评估完成")
        else:
            logger.error("❌ 本地模型评估失败")
            
        return success
        
    except Exception as e:
        logger.error(f"❌ 评估失败：{str(e)}")
        raise


def cli_eval():
    """评估命令行入口函数"""
    parser = argparse.ArgumentParser(
        description="DataFlow 评估工具 - 通过编辑算子配置文件进行评估",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用流程:
  1. 初始化配置文件：
     dataflow eval init --type api           # 生成 eval_api.py
     dataflow eval init --type local         # 生成 eval_local.py
     
  2. 编辑配置文件：
     vim eval_api.py                         # 配置模型、数据路径等参数
     
  3. 运行评估：
     dataflow eval api                       # 运行API模型评估（自动检测模型）
     dataflow eval local                     # 运行本地模型评估（自动检测模型）
     
  4. 查看配置文件：
     dataflow eval list                      # 列出所有评估配置文件

配置文件说明:
  - eval_api.py: 包含BenchDatasetEvaluator算子的完整配置，支持自动模型检测和评估
  - eval_local.py: 包含BenchDatasetEvaluator算子的完整配置，支持自动模型检测和评估
  - 用户可以直接修改这些Python文件中的参数来自定义评估配置
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="可用命令")
    
    # init 子命令
    init_parser = subparsers.add_parser(
        "init", 
        help="初始化评估配置文件",
        description="生成评估配置模板文件"
    )
    init_parser.add_argument(
        "--type", 
        choices=["api", "local"], 
        default="api",
        help="配置类型：api(使用API模型) 或 local(使用本地模型，默认：api)"
    )
    init_parser.add_argument(
        "--output", 
        default="eval_config.yaml",
        help="输出配置文件名（默认：eval_config.yaml）"
    )
    
    # api 子命令
    api_parser = subparsers.add_parser(
        "api", 
        help="使用API模型进行评估",
        description="基于配置文件使用API模型进行评估"
    )
    api_parser.add_argument(
        "--config", 
        default="eval_config.yaml",
        help="配置文件路径（默认：eval_config.yaml）"
    )
    
    # local 子命令
    local_parser = subparsers.add_parser(
        "local", 
        help="使用本地模型进行评估",
        description="基于配置文件使用本地模型进行评估"
    ) 
    local_parser.add_argument(
        "--config", 
        default="eval_config.yaml",
        help="配置文件路径（默认：eval_config.yaml）"
    )
    
    # list 子命令
    list_parser = subparsers.add_parser(
        "list", 
        help="列出当前目录的配置文件",
        description="显示当前目录下所有的评估配置文件"
    )
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    cli = DataFlowEvalCLI()
    
    try:
        if args.command == "init":
            success = cli.init_eval_file(args.type, args.output)
            if success:
                logger.info("✅ 配置文件初始化成功")
            else:
                logger.error("❌ 配置文件初始化失败")
                
        elif args.command == "api":
            logger.info("🚀 开始API模型评估...")
            success = cli.run_evaluation(args.config, "api")
            if success:
                logger.info("✅ API模型评估完成")
            else:
                logger.error("❌ API模型评估失败")
                
        elif args.command == "local":
            logger.info("🚀 开始本地模型评估...")
            success = cli.run_evaluation(args.config, "local")
            if success:
                logger.info("✅ 本地模型评估完成")
            else:
                logger.error("❌ 本地模型评估失败")
                
        elif args.command == "list":
            cli.list_configs()
            
    except KeyboardInterrupt:
        logger.info("⚠️  用户中断操作")
    except Exception as e:
        logger.error(f"❌ 执行过程中发生错误: {str(e)}")


if __name__ == "__main__":
    cli_eval()
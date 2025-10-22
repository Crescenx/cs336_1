from __future__ import annotations

import pickle
import time
from pathlib import Path

# 假设你的 BPE 训练函数位于此路径
# 请根据你的项目结构进行调整
from bpe_impl.bpe_train import train_bpe

# --- 配置区 ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

# 定义要为其训练分词器的数据集
# 脚本会为列表中的每个字符串寻找 `data/{run_str}_train.txt` 文件
RUNS = ["tinystories", "owt"]

# BPE 训练参数
# 为不同数据集设置不同的词表大小
VOCAB_SIZES = {
    "tinystories": 10000,
    "owt": 32000,
}
SPECIAL_TOKENS = ["<|endoftext|>"]

# --- 脚本主逻辑 ---

def main() -> None:
    """
    主函数，为配置区中定义的所有数据集自动训练 BPE 分词器。
    """
    # 为 BPE artifacts 创建输出目录
    output_dir = DATA_DIR / "artifacts_tokenizer"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"BPE artifacts will be saved to: {output_dir}")

    for run_str in RUNS:
        print(f"\n{'='*20} Processing dataset: {run_str.upper()} {'='*20}")

        input_path = DATA_DIR / f"{run_str}_train.txt"
        
        # 检查训练文件是否存在
        if not input_path.exists():
            print(f"⚠️ Training file not found, skipping: {input_path}")
            continue
            
        # 获取当前数据集特定的词表大小，如果未指定则使用默认值
        vocab_size = VOCAB_SIZES.get(run_str, 10000)

        print(f"Starting BPE training for '{run_str}' with vocab size {vocab_size} from source: {input_path}")
        
        start_time = time.perf_counter()
        
        # 训练 BPE，获取词汇表和合并规则
        vocab, merges = train_bpe(
            input_path=input_path,
            vocab_size=vocab_size,
            special_tokens=SPECIAL_TOKENS,
        )
        
        duration = time.perf_counter() - start_time
        print(f"✅ Training for '{run_str}' completed in {duration:.2f} seconds.")

        # --- 保存 BPE artifacts ---
        vocab_path = output_dir / f"{run_str}_vocab.pkl"
        merges_path = output_dir / f"{run_str}_merges.pkl"

        # 使用 pickle 将 vocab 对象序列化到文件
        print(f"Saving vocabulary to {vocab_path}...")
        with vocab_path.open("wb") as f:
            pickle.dump(vocab, f)

        # 使用 pickle 将 merges 对象序列化到文件
        print(f"Saving merges to {merges_path}...")
        with merges_path.open("wb") as f:
            pickle.dump(merges, f)
            
        print(f"Successfully saved artifacts for '{run_str}'.")

    print(f"\n{'='*25} All datasets processed. {'='*25}")


if __name__ == "__main__":
    main()


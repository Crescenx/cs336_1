from tokenizer.impl import Tokenizer
from pathlib import Path
import numpy as np
import time  # 导入 time 库
import os    # 导入 os 库以获取文件大小
from tqdm import tqdm # 导入 tqdm 库

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    runs = ["tinystories", "owt"]
    
    for run_str in runs:
        vocab_path = PROJECT_ROOT / "data" / "artifacts_tokenizer" / f"{run_str}_vocab.pkl"
        merges_path = PROJECT_ROOT / "data" / "artifacts_tokenizer" / f"{run_str}_merges.pkl"
        
        output_dir = PROJECT_ROOT / "data" / "tokenized"
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"--- Loading tokenizer for '{run_str}' ---")
        tokenizer = Tokenizer.from_files(vocab_path, merges_path, special_tokens=["<|endoftext|>"])

        for state in ["valid", "train"]:
            input_path = PROJECT_ROOT / "data" / f"{run_str}_{state}.txt"
            output_path = output_dir / f"{run_str}_{state}.npy"

            # 检查输入文件是否存在
            if not input_path.exists():
                print(f"⚠️ Input file not found, skipping: {input_path}")
                continue

            print(f"Processing {input_path} -> {output_path}...")
            
            # --- 代码修改部分 ---
            
            # 1. 开始计时
            start_time = time.perf_counter()
            
            # 2. 获取文件大小（以字节为单位），用于tqdm进度条
            file_size = os.path.getsize(input_path)
            
            # 用于在处理文件时更新进度条的辅助生成器
            def read_and_update_pbar(file_handle, pbar):
                for line in file_handle:
                    yield line
                    # 根据读取行的字节长度来更新进度条
                    pbar.update(len(line.encode('utf-8')))

            all_token_ids = []
            with open(input_path, "r", encoding="utf-8") as f:
                # 3. 初始化tqdm进度条
                with tqdm(total=file_size, unit="B", unit_scale=True, desc=f"Tokenizing {state}") as pbar:
                    # 将辅助生成器传给流式编码器
                    token_generator = read_and_update_pbar(f, pbar)
                    all_token_ids = list(tokenizer.encode_iterable(token_generator))

            # 4. 将 token IDs 转换为 NumPy 数组
            arr = np.array(all_token_ids, dtype=np.uint16)
            
            # 5. 保存数组
            np.save(output_path, arr)

            # 6. 结束计时并计算总时间
            end_time = time.perf_counter()
            duration = end_time - start_time
            
            # 7. 打印带有统计信息和运行时间的结果
            print(f"✅ Saved {len(arr)} tokens to {output_path} in {duration:.2f} seconds.")
            # --- 结束 ---
        print("-" * 20)
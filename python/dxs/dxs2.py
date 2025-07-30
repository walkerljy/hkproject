import numpy as np
import pandas as pd
import os

def polynomial_forecast(input_csv, output_dir="output", output_steps=10):
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取原始数据
    raw_data = pd.read_csv(input_csv, header=None).values
    
    # 验证数据有效性
    if len(raw_data) < 7:
        raise ValueError("至少需要7行数据才能进行预测")
    
    # 按每7行组成一个数据块
    block_size = 7
    results = []
    for block_idx in range(len(raw_data) // block_size):
        start_idx = block_idx * block_size
        end_idx = start_idx + block_size
        block = raw_data[start_idx:end_idx]
        
        # 为每列创建三次多项式拟合
        polynomials = []
        for col in range(5):
            x = np.arange(block_size)
            y = block[:, col]
            coeff = np.polyfit(x, y, 3)
            polynomials.append(np.poly1d(coeff))
        
        # 生成预测数据
        forecast_steps = []
        current_step = block_size
        for _ in range(output_steps):
            preds = [max(poly(current_step), 0) for poly in polynomials]
            total = sum(preds) or 1e-6
            normalized = [p/total for p in preds]
            forecast_steps.append(np.round(normalized, 4))
            current_step += 1
        
        # 构建完整时间序列
        full_timesteps = np.vstack([
            block,  # 原始数据 (7行)
            np.array(forecast_steps)  # 预测数据 (output_steps行)
        ])
        
        # 创建DataFrame
        df = pd.DataFrame(
            data=full_timesteps,
            columns=["Col1", "Col2", "Col3", "Col4", "Col5"],
            index=[f"T{t}" for t in range(block_size + output_steps)]
        )
        
        # 添加类型标记
        df["Type"] = ["Historical"]*block_size + ["Forecast"]*output_steps
        
        # 保存CSV
        output_path = os.path.join(output_dir, f"block_{block_idx+1}_forecast.csv")
        df.to_csv(output_path, float_format="%.4f")
        results.append(output_path)
    
    return results

# 使用示例
if __name__ == "__main__":
    try:
        output_files = polynomial_forecast(
            input_csv="D:/program development/hkproject/python/test_data.csv",
            output_dir="forecast_results",
            output_steps=10
        )
        
        print("\n生成的CSV文件：")
        for path in output_files:
            print(f"• {os.path.abspath(path)}")
            
    except Exception as e:
        print(f"运行错误: {str(e)}")

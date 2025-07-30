import numpy as np
import pandas as pd

def polynomial_forecast(input_csv, output_steps=10):
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
            x = np.arange(block_size)  # 时间步0-6
            y = block[:, col]
            coeff = np.polyfit(x, y, 3)
            polynomials.append(np.poly1d(coeff))
        
        # 生成后续预测
        forecast = []
        current_step = block_size  # 从第7个时间步开始预测
        for _ in range(output_steps):
            # 计算各列预测值（添加非负约束）
            preds = [max(poly(current_step), 0) for poly in polynomials]
            
            # 归一化处理
            total = sum(preds) or 1e-6  # 防止除零错误
            normalized = [p/total for p in preds]
            
            forecast.append({
                "time_step": current_step,
                "values": np.round(normalized, 4),
                "sum": round(total, 4)
            })
            current_step += 1
        
        results.append({
            "block": block_idx+1,
            "original": np.round(block, 4),
            "forecast": forecast
        })
    
    # 添加空结果保护
    if not results:
        raise ValueError("未生成任何预测结果，请检查输入数据格式")
    
    # 打印结果
    for result in results:
        print(f"\nBlock {result['block']} Forecast:")
        print("Time Step | Predicted Values (5 columns) | Sum Check")
        print("-----------------------------------------------------")
        for item in result['forecast']:
            print(f"t+{item['time_step']-6:2d} ({item['time_step']:2d}) | {item['values']} | {item['sum']:.4f}")
    
    return results


# 使用示例
output = polynomial_forecast("D:/program development/hkproject/python/test_data.csv", 
                           output_steps=10)

# 验证第一个预测块的最后一个预测
sample_pred = output[0]['forecast'][-1]['values']
print("\nFinal prediction example:", sample_pred)

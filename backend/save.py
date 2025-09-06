import os
import numpy as np
import re

# -------------------------- 新增：按“人员+睡姿”导出数据（适配前端选择） --------------------------
def export_person_posture_data(X_raw, y_raw, original_person_dirs, data_root_dir, export_root_dir):
    """
    按“人员+睡姿”导出数据：每个人员对应一个文件夹，每个文件夹下存该人员的21种睡姿样本
    参数说明：
    - X_raw：加载的全量原始样本（40×26压力矩阵）
    - y_raw：加载的全量原始标签（1-21，未编码）
    - original_person_dirs：加载数据时记录的“原始人员文件夹名”（如[xby0902, zhangsan0801]）
    - data_root_dir：原始数据根目录（即你的DATA_DIR="data"）
    - export_root_dir：导出数据的根目录（对应backend/model_output）
    """
    # 1. 先为每个样本标记“所属人员”（建立样本索引→人员名的映射）
    sample_to_person = []  # 存储每个样本对应的人员名
    for person_dir in original_person_dirs:
        # 读取该人员文件夹下的所有txt文件，计算每个文件的样本数
        person_dir_path = os.path.join(data_root_dir, person_dir)
        for filename in os.listdir(person_dir_path):
            if not filename.endswith(".txt") or not re.search(r"(\d+)\.txt$", filename):
                continue
            # 计算该txt文件的样本数（40行1个样本）
            file_path = os.path.join(person_dir_path, filename)
            try:
                full_data = np.loadtxt(file_path, delimiter=",", dtype=np.float32)
                total_samples = full_data.shape[0] // 40
                # 为该文件的每个样本标记所属人员
                sample_to_person.extend([person_dir] * total_samples)
            except Exception as e:
                print(f"计算{person_dir}/{filename}样本数失败：{str(e)}")
                continue
    
    # 校验：样本数与人员标记数一致（确保映射正确）
    if len(sample_to_person) != len(X_raw):
        raise ValueError(f"样本数（{len(X_raw)}）与人员标记数（{len(sample_to_person)}）不匹配，请检查原始数据加载逻辑")
    
    # 2. 按“人员”分组，导出该人员的所有睡姿样本
    unique_persons = list(set(sample_to_person))  # 去重，得到所有人员名
    for person in unique_persons:
        print(f"\n开始导出人员【{person}】的睡姿数据...")
        # 创建该人员的导出文件夹（如model_output/xby0902）
        person_export_dir = os.path.join(export_root_dir, person)
        os.makedirs(person_export_dir, exist_ok=True)
        
        # 3. 遍历21种睡姿，为该人员找每种睡姿的样本
        for posture_id in range(1, 22):
            # 找到“该人员”且“睡姿为posture_id”的所有样本索引
            target_sample_indices = [
                idx for idx in range(len(X_raw))
                if sample_to_person[idx] == person and y_raw[idx] == posture_id
            ]
            
            if len(target_sample_indices) == 0:
                print(f"人员【{person}】无睡姿{posture_id}的样本，跳过")
                continue
            
            # 取第一个样本（也可修改为取多个，按需求调整）
            target_sample_idx = target_sample_indices[0]
            pressure_matrix = X_raw[target_sample_idx]  # 该人员该睡姿的40×26压力矩阵
            
            # 保存为npy文件（命名格式：posture_1.npy~posture_21.npy）
            save_path = os.path.join(person_export_dir, f"posture_{posture_id}.npy")
            np.save(save_path, pressure_matrix)
            print(f"已保存：{save_path}")
    
    print(f"\n所有人员数据导出完成！导出根目录：{export_root_dir}")



if __name__ == "__main__":
    # 新增：记录加载数据时的原始人员文件夹名（如xby0902）
    original_person_dirs = []  # 存储所有参与数据加载的人员文件夹名
    X_raw = []
    y_raw = []
    data_root_dir = 'data'  # 即你的DATA_DIR="data"
    
    # 复制原load_dataset函数的核心逻辑，但添加“记录人员文件夹名”
    for person_dir in os.listdir(data_root_dir):
        person_dir_path = os.path.join(data_root_dir, person_dir)
        if not os.path.isdir(person_dir_path):
            continue
        original_person_dirs.append(person_dir)  # 关键：记录当前加载的人员文件夹名
        print(f"正在读取 {person_dir} 的数据...")
        
        for filename in os.listdir(person_dir_path):
            if not filename.endswith(".txt"):
                continue
            match = re.search(r"(\d+)\.txt$", filename)
            if not match:
                print(f"跳过非目标文件：{filename}（未匹配数字后缀）")
                continue
            posture_idx = int(match.group(1))
            file_path = os.path.join(person_dir_path, filename)
            
            try:
                full_data = np.loadtxt(file_path, delimiter=",", dtype=np.float32)
                total_samples = full_data.shape[0] // 40
                if total_samples == 0:
                    print(f"跳过数据不足文件：{filename}（不足40行）")
                    continue
                for i in range(total_samples):
                    start_idx = i * 40
                    end_idx = start_idx + 40
                    sample = full_data[start_idx:end_idx, :]
                    if sample.shape[1] != 26:
                        print(f"跳过列数错误文件：{filename}（实际{sample.shape[1]}列，需26列）")
                        break
                    X_raw.append(sample)
                    y_raw.append(posture_idx)
            except Exception as e:
                print(f"读取文件失败：{filename}，错误：{str(e)}")
                continue
    
    # 转换为numpy数组（与原load_dataset一致）
    X_raw = np.array(X_raw, dtype=np.float32)
    y_raw = np.array(y_raw, dtype=np.int32)
    export_root_dir = os.path.join(os.path.dirname(__file__), "model_output")
    # 调用导出函数（传入全量原始数据、原始人员名、原始数据根目录、导出根目录）
    export_person_posture_data(X_raw, y_raw, original_person_dirs, data_root_dir, export_root_dir)
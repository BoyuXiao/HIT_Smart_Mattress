import numpy as np
import os
import re
from skimage.transform import resize
from skimage.morphology import closing, opening, disk
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pygad
import joblib
import warnings
import pickle
warnings.filterwarnings("ignore")

PERSON_LABEL_MAP = {}
PRESSURE_MAP_SHAPE = (40, 26)  
DATA_DIR = "data"  
POSTURE_LABEL_MAP = {i: f"睡姿{i}" for i in range(1, 22)}  

GA_NUM_GENERATIONS = 5    
GA_NUM_PARENTS_MATING = 4 
GA_SOL_PER_POP = 8        
GA_MUTATION_PERCENT = 20  

VAL_SIZE = 0.2  
CV_FOLDS = 5              
TEST_SIZE = 0.3            


def load_dataset(data_root_dir):
    X_raw = []
    y_posture = []  # 睡姿标签
    y_person = []   # 人员标签
    person_ids = []
    person_idx = 0
    
    for person_dir in os.listdir(data_root_dir):
        person_dir_path = os.path.join(data_root_dir, person_dir)
        if not os.path.isdir(person_dir_path):
            continue
            
        # 为每个人分配唯一ID
        PERSON_LABEL_MAP[person_idx] = person_dir
        person_ids.append(person_idx)
        
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
                    y_posture.append(posture_idx)
                    y_person.append(person_idx)  # 添加人员标签
            except Exception as e:
                print(f"读取文件失败：{filename}，错误：{str(e)}")
                continue
        
        person_idx += 1
    
    X_raw = np.array(X_raw, dtype=np.float32)
    y_posture = np.array(y_posture, dtype=np.int32)
    y_person = np.array(y_person, dtype=np.int32)
    
    # 保存人员标签映射
    # with open('backend/models/person_label_map.pkl', 'wb') as f:
    with open('models/person_label_map.pkl', 'wb') as f:
        pickle.dump(PERSON_LABEL_MAP, f)
    
    print(f"数据集加载完成：共{len(X_raw)}个样本，{len(np.unique(y_posture))}种睡姿，{len(np.unique(y_person))}个人员")
    return X_raw, y_posture, y_person


def preprocess_pressure_map(pressure_map):
    # 确保输入是数值类型（避免布尔类型错误）
    pressure_map = pressure_map.astype(np.float32)
    
    # 形态学操作（闭运算+开运算）
    selem = disk(2) 
    closed_image = closing(pressure_map, selem)
    opened_image = opening(closed_image, selem)

    # 归一化（避免除以0）
    min_val = np.min(opened_image)
    max_val = np.max(opened_image)
    if max_val - min_val < 1e-6:
        normalized_image = opened_image
    else:
        normalized_image = (opened_image - min_val) / (max_val - min_val)

    # 展平为1D特征（适配SVM）
    return normalized_image.flatten()



train_sub_flat, val_sub_flat, y_train_sub, y_val_sub = None, None, None, None

def fitness_func(ga_instance, solution, solution_idx):
    """
    新适应度函数：用子训练集训练，独立验证集评估（替代K折CV）
    """
    global train_sub_flat, val_sub_flat, y_train_sub, y_val_sub
    C = solution[0]
    gamma = solution[1]
    
    # 优化SVM参数：增大cache_size（减少缓存交换时间，提速）
    svm = SVC(
        C=C, gamma=gamma, kernel="rbf", 
        random_state=42,
        cache_size=1000  # 原默认200MB，增大至1000MB（需根据内存调整，最大不超过可用内存）
    )
    
    # 仅用子训练集训练，验证集评估（1次训练+1次预测，速度快）
    svm.fit(train_sub_flat, y_train_sub)
    val_acc = svm.score(val_sub_flat, y_val_sub)  # 验证集准确率作为适应度
    
    # 打印当前超参数的验证结果（实时观察）
    print(f"  第{ga_instance.generations_completed}代-个体{solution_idx}：C={C:.4f}, gamma={gamma:.4f}，验证准确率={val_acc:.4f}")
    return val_acc

def optimize_svm_hyperparams(X_train, y_train):
    global train_sub_flat, val_sub_flat, y_train_sub, y_val_sub
    
    # 步骤1：先对训练集做预处理（展平为1D特征）
    X_train_flat = np.array([preprocess_pressure_map(map_) for map_ in X_train])
    
    # 步骤2：拆分“子训练集”和“独立验证集”（分层抽样，保持类别分布）
    train_sub_flat, val_sub_flat, y_train_sub, y_val_sub = train_test_split(
        X_train_flat, y_train,
        test_size=VAL_SIZE,  # 从训练集中划20%作为验证集
        random_state=42,
        stratify=y_train     # 关键：保持睡姿类别分布一致
    )
    print(f"\n超参数优化数据拆分完成：")
    print(f"  子训练集大小：{len(train_sub_flat)}，独立验证集大小：{len(val_sub_flat)}")
    
    # 超参数搜索空间（保持不变）
    gene_space = [
        {"low": 0.01, "high": 100},  # C的范围
        {"low": 0.01, "high": 100}   # gamma的范围
    ]
    
    # 每代进度回调（保持不变）
    def on_generation(ga_instance):
        best_sol, best_acc, _ = ga_instance.best_solution()
        print(f"\n第 {ga_instance.generations_completed} 代完成！当前最优适应度：{best_acc:.4f}")
    
    # 初始化GA实例（参数微调：减少种群规模）
    ga_instance = pygad.GA(
        num_generations=GA_NUM_GENERATIONS,
        num_parents_mating=GA_NUM_PARENTS_MATING,
        fitness_func=fitness_func,
        sol_per_pop=GA_SOL_PER_POP,  # 种群规模从10减至8，减少每代计算量
        num_genes=2,
        gene_space=gene_space,
        parent_selection_type="sss",
        keep_parents=1,
        crossover_type="single_point",
        mutation_type="random",
        mutation_percent_genes=GA_MUTATION_PERCENT,
        random_seed=42,
        on_generation=on_generation
    )
    
    # 运行GA优化
    print("\n开始用遗传算法优化SVM超参数...")
    ga_instance.run()
    
    # 获取最优结果
    best_solution, best_fitness, _ = ga_instance.best_solution()
    best_C, best_gamma = best_solution[0], best_solution[1]
    print(f"\nGA优化完成！最优超参数：C={best_C:.4f}, gamma={best_gamma:.4f}")
    print(f"最优验证集准确率：{best_fitness:.4f}")
    
    return best_C, best_gamma


def train_and_evaluate_svm_p(best_C, best_gamma, X_train, X_test, y_train, y_test):
    X_train_flat = np.array([preprocess_pressure_map(map_) for map_ in X_train])
    X_test_flat = np.array([preprocess_pressure_map(map_) for map_ in X_test])
    
    # 同样增大SVM的cache_size，提速
    svm_model = SVC(
        C=best_C, gamma=best_gamma, kernel="rbf", 
        random_state=42, cache_size=1000
    )
    print("\n用最优超参数训练最终SVM模型...")
    svm_model.fit(X_train_flat, y_train)
    
    # 评估
    y_pred = svm_model.predict(X_test_flat)
    accuracy = accuracy_score(y_test, y_pred)
    conf_mat = confusion_matrix(y_test, y_pred)
    class_report = classification_report(
        y_test, y_pred,
        target_names=[POSTURE_LABEL_MAP[i+1] for i in range(21)],
        digits=4
    )
    
    print(f"\n=== 模型评估结果 ===")
    print(f"测试集准确率：{accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"\n混淆矩阵（行：真实标签，列：预测标签）：")
    print(conf_mat)
    print(f"\n分类报告（精确率/召回率/F1）：")
    print(class_report)
    
    return svm_model, accuracy, conf_mat


def train_and_evaluate_svm_per(best_C, best_gamma, X_train, X_test, y_train, y_test, label_map):
    """
    训练并评估SVM模型，支持动态标签映射
    
    参数:
        best_C: SVM的C参数
        best_gamma: SVM的gamma参数
        X_train, X_test: 训练集和测试集特征
        y_train, y_test: 训练集和测试集标签
        label_map: 标签映射字典（键为数字标签，值为实际类别名称）
        model_name: 模型名称（用于输出信息区分）
    """
    # 数据预处理（展平压力矩阵）
    X_train_flat = np.array([preprocess_pressure_map(map_) for map_ in X_train])
    X_test_flat = np.array([preprocess_pressure_map(map_) for map_ in X_test])
    
    # 初始化SVM模型
    svm_model = SVC(
        C=best_C, gamma=best_gamma, kernel="rbf", 
        random_state=42, cache_size=1000
    )
    svm_model.fit(X_train_flat, y_train)
    
    # 模型评估
    y_pred = svm_model.predict(X_test_flat)
    accuracy = accuracy_score(y_test, y_pred)
    conf_mat = confusion_matrix(y_test, y_pred)
    
    # 生成分类报告（动态获取类别名称）
    # 提取所有唯一标签并排序
    unique_labels = sorted(np.unique(np.concatenate([y_test, y_pred])))
    target_names = [label_map[label] for label in unique_labels]
    
    class_report = classification_report(
        y_test, y_pred,
        target_names=target_names,
        digits=4
    )
    
    # 输出评估结果
    print(f"测试集准确率：{accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"\n混淆矩阵（行：真实标签，列：预测标签）：")
    print(conf_mat)
    print(f"\n分类报告（精确率/召回率/F1）：")
    print(class_report)
    
    return svm_model, accuracy, conf_mat

def predict_single_pressure_map(model, pressure_map):
    """
    预测单张压力图的睡姿类别
    :param model: 加载好的SVM模型
    :param pressure_map: 40x26的压力矩阵
    :return: 预测的睡姿ID（1-21）
    """
    # 预处理（复用现有函数）
    processed = preprocess_pressure_map(pressure_map)
    # 预测（模型输出0-20，转换为1-21）
    prediction = model.predict([processed])[0]
    return int(prediction) + 1

def predict_single_person(model, pressure_map):
    """预测单张压力图的人员类别"""
    # 预处理（复用现有函数）
    processed = preprocess_pressure_map(pressure_map)
    # 预测
    prediction = model.predict([processed])[0]
    
    # 加载人员标签映射
    # with open('backend/models/person_label_map.pkl', 'rb') as f:
    with open('models/person_label_map.pkl', 'rb') as f:
        person_map = pickle.load(f)
    
    return person_map.get(int(prediction), "未知人员")

def load_trained_model(model_path="svm_model.pkl"):
    """加载保存的SVM模型"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件 {model_path} 不存在，请先训练模型")
    return joblib.load(model_path)

if __name__ == "__main__":
    # 1. 加载全量数据，同时获取睡姿和人员标签
    X_raw, y_posture_raw, y_person_raw = load_dataset(DATA_DIR)
    
    # 2. 标签编码
    y_posture_encoded = y_posture_raw - 1  # 睡姿1-21→0-20
    y_person_encoded = y_person_raw  # 人员标签已经是0开始的索引
    
    # 3. 拆分全量训练集和测试集（7:3）
    # 睡姿模型
    X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(
        X_raw, y_posture_encoded,
        test_size=TEST_SIZE,
        random_state=42,
        stratify=y_posture_encoded,
        shuffle=True
    )
    
    # 人员模型
    X_train_per, X_test_per, y_train_per, y_test_per = train_test_split(
        X_raw, y_person_encoded,
        test_size=TEST_SIZE,
        random_state=42,
        stratify=y_person_encoded,
        shuffle=True
    )
    
    print(f"\n全量数据拆分完成：训练集大小：{len(X_train_p)}，测试集大小：{len(X_test_p)}")
    
    # 4. 训练睡姿模型
    print("\n===== 训练睡姿模型 =====")
    best_C_p, best_gamma_p = 16.5350, 0.5622  # 可以保留现有优化值或重新优化
    # best_C_p, best_gamma_p = optimize_svm_hyperparams(X_train_p, y_train_p)
    posture_model, test_accuracy_p, conf_mat_p = train_and_evaluate_svm_p(
        best_C_p, best_gamma_p, X_train_p, X_test_p, y_train_p, y_test_p
    )
    print(f"\n睡姿模型最终测试集准确率：{test_accuracy_p:.4f} ({test_accuracy_p*100:.2f}%)")
    
    # 5. 训练人员模型
    print("\n===== 训练人员模型 =====")
    best_C_per, best_gamma_per= 92.1882, 15.6079
    # best_C_per, best_gamma_per = optimize_svm_hyperparams(X_train_per, y_train_per)
    person_model, test_accuracy_per, conf_mat_per = train_and_evaluate_svm_per(
        best_C_per, best_gamma_per, X_train_per, X_test_per, y_train_per, y_test_per, PERSON_LABEL_MAP
    )
    print(f"\n人员模型最终测试集准确率：{test_accuracy_per:.4f} ({test_accuracy_per*100:.2f}%)")
    
    # 保存模型
    # os.makedirs('backend/models', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    # joblib.dump(posture_model, "backend/models/svm_posture_model.pkl")
    joblib.dump(posture_model, "models/svm_posture_model.pkl")
    # joblib.dump(person_model, "backend/models/svm_person_model.pkl")
    joblib.dump(person_model, "models/svm_person_model.pkl")
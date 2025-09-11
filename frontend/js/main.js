// main.js (只修改与热力图获取相关的部分)
// DOM元素
const personSelect = document.getElementById('personSelect');
const postureSelect = document.getElementById('postureSelect');
const analyzeBtn = document.getElementById('analyzeBtn');


// 模块元素
const heatmapModule = {
    image: document.getElementById('heatmapImage'),
    placeholder: document.getElementById('heatmapPlaceholder'),
    status: document.getElementById('heatmapStatus')
};

const regionModule = {
    canvas: document.getElementById('regionCanvas'),
    placeholder: document.getElementById('regionPlaceholder'),
    status: document.getElementById('regionStatus')
};
const rctx = regionModule.canvas ? regionModule.canvas.getContext('2d') : null;

const resultModule = {
    posture: document.getElementById('postureResult'),
    person: document.getElementById('personResult'),
    status: document.getElementById('resultStatus')
};

// 1. 加载人员列表
async function loadPeople() {
    try {
        const res = await fetch(`${API_BASE}/api/people`);
        if (!res.ok) throw new Error(`HTTP错误: ${res.status}`);
        
        const data = await res.json();
        if (data.code === 200) {
            data.data.forEach(person => {
                const option = document.createElement('option');
                option.value = person;
                option.textContent = person;
                personSelect.appendChild(option);
            });
        } else {
            showError('加载人员列表失败', data.msg);
        }
    } catch (err) {
        showError('加载人员列表出错', err.message);
        console.error('人员列表加载错误:', err);
    }
}

// 2. 加载姿势下拉框（1-21）
function loadPostures() {
    postureSelect.innerHTML = '<option value="">-- 请选择姿势 --</option>';
    for (let i = 1; i <= 21; i++) {
        const option = document.createElement('option');
        option.value = i;
        option.textContent = `姿势${i}`;
        postureSelect.appendChild(option);
    }
}

// 3. 显示加载状态
function showLoading() {
    // 重置所有模块
    resetModules();
    
    // 显示加载状态
    heatmapModule.status.innerHTML = '<span class="status-loading"><i class="fas fa-spinner fa-spin"></i> 生成中</span>';
    regionModule.status.innerHTML = '<span class="status-loading"><i class="fas fa-spinner fa-spin"></i> 生成中</span>';
    resultModule.status.innerHTML = '<span class="status-loading"><i class="fas fa-spinner fa-spin"></i> 分析中</span>';
    
    // 显示加载动画（创建临时加载层）
    [heatmapModule, regionModule].forEach(module => {
        const loadingDiv = document.createElement('div');
        loadingDiv.className = 'loading-overlay';
        loadingDiv.innerHTML = `
            <div class="loading-spinner"></div>
            <p>正在处理，请稍候...</p>
        `;
        loadingDiv.id = module === heatmapModule ? 'heatmapLoading' : 'regionLoading';
        const parent = module.canvas ? module.canvas.parentNode : module.image.parentNode;
        parent.appendChild(loadingDiv);
        loadingDiv.style.display = 'flex';
    });
    
    // 结果模块显示加载中
    resultModule.posture.innerHTML = '<i class="fas fa-spinner fa-spin"></i> 分析中...';
    resultModule.person.innerHTML = '<i class="fas fa-spinner fa-spin"></i> 分析中...';
}

// 4. 重置模块状态
function resetModules() {
    // 移除加载层
    const loadings = document.querySelectorAll('.loading-overlay');
    loadings.forEach(loading => loading.remove());
    
    // 重置图片
    heatmapModule.image.src = '';
    heatmapModule.image.style.display = 'none';
    heatmapModule.placeholder.style.display = 'block';
    
    if (regionModule.canvas) {
        regionModule.canvas.style.display = 'none';
        if (rctx) { rctx.clearRect(0,0,regionModule.canvas.width, regionModule.canvas.height); }
    }
    regionModule.placeholder.style.display = 'block';
}

// 5. 显示错误信息
function showError(title, message) {
    alert(`${title}: ${message}`);
    
    // 重置状态为错误
    heatmapModule.status.innerHTML = '<span class="status-error">加载失败</span>';
    regionModule.status.innerHTML = '<span class="status-error">加载失败</span>';
    resultModule.status.innerHTML = '<span class="status-error">分析失败</span>';
}

// 6. 开始分析流程
async function startAnalysis() {
    const person = personSelect.value;
    const postureId = postureSelect.value;
    
    if (!person || !postureId) {
        alert('请选择人员和姿势后再进行分析');
        return;
    }
    
    // 显示加载状态
    showLoading();
    
    try {
        // 步骤1: 生成热力图图片
        const heatmapRes = await fetch(`${API_BASE}/api/generate_heatmap_image?person=${person}&posture_id=${postureId}`);
        if (!heatmapRes.ok) throw new Error(`生成热力图失败: ${heatmapRes.status}`);
        const heatmapData = await heatmapRes.json();
        console.log('热力图接口返回数据:', heatmapData);
        if (heatmapData.code !== 200) {
            throw new Error(heatmapData.msg || '生成热力图失败');
        }
        // 加载生成的热力图
        heatmapModule.image.src = heatmapData.data.image_url; 
        heatmapModule.image.onload = () => {
            heatmapModule.image.style.display = 'block';
            heatmapModule.placeholder.style.display = 'none';
            heatmapModule.status.innerHTML = '<span class="status-ready">已生成</span>';
            document.getElementById('heatmapLoading')?.remove();
        };
        
        // 步骤2: 获取压力矩阵数据（用于预测）
        const pressureRes = await fetch(`${API_BASE}/api/heatmap?person=${person}&posture_id=${postureId}`);
        if (!pressureRes.ok) throw new Error(`获取压力数据失败: ${pressureRes.status}`);
        const pressureData = await pressureRes.json();
        if (pressureData.code !== 200) {
            throw new Error(pressureData.msg || '获取压力数据失败');
        }
        
        // 步骤3: 调用预测接口
        const predictRes = await fetch(`${API_BASE}/api/predict_posture`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ matrix: pressureData.data.matrix })
        });
        if (!predictRes.ok) throw new Error(`预测失败: ${predictRes.status}`);
        const predictResult = await predictRes.json();
        if (predictResult.code !== 200) {
            throw new Error(predictResult.msg || '预测失败');
        }
        
        const predictPersonRes = await fetch(`${API_BASE}/api/predict_person`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ matrix: pressureData.data.matrix })
        });
        if (!predictPersonRes.ok) throw new Error(`人员预测失败: ${predictPersonRes.status}`);
        const predictPersonResult = await predictPersonRes.json();
        if (predictPersonResult.code !== 200) {
            throw new Error(predictPersonResult.msg || '人员预测失败');
        }
        
        // 步骤4: 区域划分
        const regionRes = await fetch(`${API_BASE}/api/predict_regions`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ matrix: pressureData.data.matrix })
        });
        if (!regionRes.ok) throw new Error(`区域划分失败: ${regionRes.status}`);
        const regionData = await regionRes.json();
        if (regionData.code !== 200) throw new Error(regionData.msg || '区域划分失败');

        await drawRegions(regionData.data.boxes);
        
        // 步骤5: 显示预测结果
        resultModule.posture.textContent = `姿势${predictResult.data.predicted_posture_id}`;
        resultModule.person.textContent = predictPersonResult.data.predicted_person;
        resultModule.status.innerHTML = '<span class="status-ready">分析完成</span>';
        
    } catch (err) {
        showError('分析过程出错', err.message);
        console.error('分析错误:', err);
    }
}

async function drawRegions(boxes) {
    if (!regionModule.canvas || !rctx) return;
    // 使用已生成的热力图作为背景
    const bg = new Image();
    bg.src = document.getElementById('heatmapImage').src;
    await new Promise((resolve) => { bg.onload = resolve; });
    regionModule.canvas.width = bg.width;
    regionModule.canvas.height = bg.height;
    rctx.clearRect(0,0,regionModule.canvas.width, regionModule.canvas.height);
    rctx.drawImage(bg, 0, 0, regionModule.canvas.width, regionModule.canvas.height);

    const sx = regionModule.canvas.width / 26.0;
    const sy = regionModule.canvas.height / 40.0;
    rctx.lineWidth = 2;
    rctx.strokeStyle = 'red';
    boxes.forEach(([x1,y1,x2,y2]) => {
        const rx = Math.round(x1 * sx);
        const ry = Math.round(y1 * sy);
        const rw = Math.round((x2 - x1) * sx);
        const rh = Math.round((y2 - y1) * sy);
        rctx.strokeRect(rx, ry, rw, rh);
    });
    regionModule.canvas.style.display = 'block';
    regionModule.placeholder.style.display = 'none';
    regionModule.status.innerHTML = '<span class="status-ready">已生成</span>';
    document.getElementById('regionLoading')?.remove();
}

// 绑定事件
analyzeBtn.addEventListener('click', startAnalysis);
personSelect.addEventListener('change', loadPostures);

// 页面加载时执行
window.onload = async () => {
    await loadPeople();
    loadPostures();
};
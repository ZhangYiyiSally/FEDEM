SetFactory("OpenCASCADE");
Geometry.OCCScaling=0.001;
Merge "e4-2.step";
//+
Physical Surface("AllSurface", 37) = {13, 9, 6, 7, 5, 11, 12, 8, 10, 4, 3, 2, 14, 1};
//+
Physical Surface("InSurface", 38) = {5, 7, 6, 9, 13, 11, 10, 8, 12};
//+
Physical Surface("OutSurface", 39) = {2, 3, 4};
//+
Physical Surface("Symmetry", 40) = {1, 14};
//+
Physical Volume("AllVolume", 41) = {1};

// 设置基础参数
Mesh.Algorithm = 6;       // 2D推荐使用Frontal-Delaunay算法
Mesh.Smoothing = 15;      // 增加光顺迭代次数
Mesh.AnisoMax = 2;       // 限制各向异性比

// 定义曲率场（自动识别所有曲线）
Field[1] = Curvature;
Field[1].Delta = 0.005;            // 曲率敏感度阈值

// 设置尺寸渐变控制
Field[2] = Threshold;
Field[2].InField = 1;
Field[2].LcMin = 0.005;    // 保持最小尺寸
Field[2].LcMax = 0.03;     // 最大尺寸
Field[2].DistMin = 0.06;   // 距离≤DistMin时保持LcMin
Field[2].DistMax = 0.1;   // 距离≥DistMax时保持LcMax

Background Field = 2;       // 激活复合场

// 高级选项：针对复杂曲线的强化控制
Mesh.MeshSizeExtendFromBoundary = 0;  // 禁用边界尺寸传播
Mesh.MeshSizeFromPoints = 0;          // 禁用点尺寸影响
Mesh.MeshSizeFromCurvature = 50;      // 曲率自适应强度（0-100）

// 生成网格
Mesh 3;  // 或 Mesh 3 生成三维网格
Save "e4-2_mesh.msh";

"""
ZMX文件高级公差分析系统
支持多视场点Fringe泽尼克系数分析，包含偏心、倾斜、间隔、折射率误差、面型误差（泽尼克定义）
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import zospy as zp
from zospy.analyses import operands


class AdvancedToleranceAnalyzer:
    """高级公差分析器，支持多视场点和多种误差类型"""
    
    def __init__(self, zos_path: str = None):
        """
        初始化分析器
        
        Args:
            zos_path: OpticStudio安装路径，None则自动检测
        """
        self.zos = zp.ZOS()
        self.zos.connect_as_extension(zos_path)
        self.system = None
        self.lens_data = None
        self.fields = []
        self.wavelengths = []
        
    def load_system(self, zmx_file_path: str) -> bool:
        """
        加载ZMX文件
        
        Args:
            zmx_file_path: ZMX文件完整路径
            
        Returns:
            bool: 加载成功与否
        """
        try:
            self.system = self.zos.PrimarySystem
            self.system.loadFile(zmx_file_path)
            self.lens_data = self.system.LDE
            self._update_system_info()
            print(f"成功加载系统：{zmx_file_path}")
            print(f"表面数量：{self.lens_data.NumberOfSurfaces}")
            print(f"视场点数量：{len(self.fields)}")
            print(f"波长数量：{len(self.wavelengths)}")
            return True
        except Exception as e:
            print(f"加载系统失败：{e}")
            return False
    
    def _update_system_info(self):
        """更新系统信息（视场点、波长等）"""
        # 获取视场点
        field_data = self.system.SystemData.Fields
        self.fields = []
        for i in range(1, field_data.NumberOfFields + 1):
            field = field_data.GetField(i)
            self.fields.append({
                'index': i,
                'x': field.X,
                'y': field.Y,
                'type': field.FieldType
            })
        
        # 获取波长
        wave_data = self.system.SystemData.Wavelengths
        self.wavelengths = []
        for i in range(1, wave_data.NumberOfWavelengths + 1):
            wave = wave_data.GetWavelength(i)
            self.wavelengths.append({
                'index': i,
                'value': wave.Wave,
                'weight': wave.Weight
            })
    
    def get_lens_surfaces(self) -> List[Dict]:
        """
        获取所有镜片表面信息
        
        Returns:
            List[Dict]: 镜片表面列表，包含表面编号、类型、材料等信息
        """
        lens_surfaces = []
        for i in range(1, self.lens_data.NumberOfSurfaces + 1):
            surface = self.lens_data.GetSurfaceAt(i)
            surf_data = {
                'index': i,
                'name': surface.Name,
                'type': surface.Type,
                'material': surface.Material,
                'radius': surface.Radius,
                'thickness': surface.Thickness,
                'semi_diameter': surface.SemiDiameter,
                'is_standard': surface.Type == "Standard"
            }
            
            # 判断是否为镜片表面（有厚度且为标准面）
            if surf_data['is_standard'] and surf_data['thickness'] != 0:
                lens_surfaces.append(surf_data)
        
        return lens_surfaces
    
    def apply_decenter(self, surface_index: int, decenter_x: float, decenter_y: float):
        """
        应用偏心操作
        
        Args:
            surface_index: 表面编号
            decenter_x: X方向偏心量 (mm)
            decenter_y: Y方向偏心量 (mm)
        """
        surface = self.lens_data.GetSurfaceAt(surface_index)
        surface.DecenterX = decenter_x
        surface.DecenterY = decenter_y
        print(f"表面 {surface_index}: 应用偏心 ({decenter_x:.4f}, {decenter_y:.4f}) mm")
    
    def apply_tilt(self, surface_index: int, tilt_x: float, tilt_y: float, tilt_z: float):
        """
        应用倾斜操作
        
        Args:
            surface_index: 表面编号
            tilt_x: X方向倾斜角度 (度)
            tilt_y: Y方向倾斜角度 (度)
            tilt_z: Z方向倾斜角度 (度)
        """
        surface = self.lens_data.GetSurfaceAt(surface_index)
        surface.TiltAboutX = tilt_x
        surface.TiltAboutY = tilt_y
        surface.TiltAboutZ = tilt_z
        print(f"表面 {surface_index}: 应用倾斜 ({tilt_x:.4f}, {tilt_y:.4f}, {tilt_z:.4f}) 度")
    
    def apply_thickness_error(self, surface_index: int, thickness_error: float):
        """
        应用间隔（厚度）误差
        
        Args:
            surface_index: 表面编号
            thickness_error: 厚度误差量 (mm)
        """
        surface = self.lens_data.GetSurfaceAt(surface_index)
        original_thickness = surface.Thickness
        surface.Thickness = original_thickness + thickness_error
        print(f"表面 {surface_index}: 厚度 {original_thickness:.4f} -> {surface.Thickness:.4f} mm (误差: {thickness_error:.4f})")
    
    def apply_refractive_index_error(self, surface_index: int, dn: float):
        """
        应用折射率误差
        
        Args:
            surface_index: 表面编号
            dn: 折射率误差量
        """
        surface = self.lens_data.GetSurfaceAt(surface_index)
        material = surface.Material
        
        if material and material != "AIR":
            # 获取当前折射率
            wave_index = self.wavelengths[0]['index']  # 使用主波长
            current_n = self.system.SystemData.MaterialCatalog.GetMaterial(material).GetIndex(wave_index)
            
            # 注意：ZOSPy中直接修改材料折射率需要通过替代材料或参数调整
            # 这里使用近似方法：通过温度压力或直接修改（如果支持）
            print(f"表面 {surface_index}: 材料 {material}, 折射率误差: {dn:.6f}")
            print("  注：折射率误差需要通过材料替换或参数优化实现")
        else:
            print(f"表面 {surface_index}: 空气或无材料，跳过折射率误差")
    
    def apply_zernike_surface_error(self, surface_index: int, 
                                   zernike_coeffs: Dict[int, float],
                                   norm_radius: float = None):
        """
        应用泽尼克面型误差
        
        Args:
            surface_index: 表面编号
            zernike_coeffs: 泽尼克系数字典 {term_index: coefficient}
                          term_index: 泽尼克项序号 (1-based, Fringe标准)
                          coefficient: 系数值 (微米)
            norm_radius: 归一化半径 (mm), None则使用表面半口径
        """
        surface = self.lens_data.GetSurfaceAt(surface_index)
        
        # 转换为Zernike Standard Sag面型
        original_type = surface.Type
        surface.Type = "Zernike Standard Sag"
        
        # 设置归一化半径
        if norm_radius is None:
            norm_radius = surface.SemiDiameter
        surface.ParameterData.GetParameterAt(1).Value = norm_radius  # Norm Radius
        
        # 设置泽尼克系数 (Fringe标准，从第2个参数开始)
        for term_idx, coeff in zernike_coeffs.items():
            param_idx = term_idx + 1  # Fringe泽尼克从参数2开始
            if param_idx <= surface.ParameterData.NumberOfParameters:
                surface.ParameterData.GetParameterAt(param_idx).Value = coeff * 1e-3  # 转换为mm
                print(f"  泽尼克项 Z{term_idx}: {coeff:.4f} μm")
        
        print(f"表面 {surface_index}: 应用泽尼克面型误差 ({len(zernike_coeffs)} 项)")
    
    def calculate_zernike_coefficients(self, field_index: int = 1, 
                                      wavelength_index: int = 1,
                                      num_terms: int = 37) -> Dict:
        """
        计算指定视场和波长的Fringe泽尼克系数
        
        Args:
            field_index: 视场点索引 (1-based)
            wavelength_index: 波长索引 (1-based)
            num_terms: 计算的泽尼克项数 (最多37项)
            
        Returns:
            Dict: 泽尼克系数结果
        """
        try:
            # 创建泽尼克系数分析
            zernike_analysis = self.system.Analyses.New_Analysis_SettingsFirst("Zernike Coefficients")
            
            # 配置分析设置
            settings = zernike_analysis.GetSettings()
            settings.FieldNumber = field_index
            settings.Wavelength = wavelength_index
            settings.UsePolarization = False
            settings.NumTerms = num_terms
            
            # 运行分析
            zernike_analysis.ApplyAndWaitForCompletion()
            
            # 获取结果
            results = zernike_analysis.GetResults()
            data_grid = results.GetDataGrid(0)
            
            zernike_results = {
                'field_index': field_index,
                'wavelength_index': wavelength_index,
                'coefficients': {},
                'rms_wavefront': 0.0,
                'pv_wavefront': 0.0
            }
            
            # 解析泽尼克系数
            for row in range(data_grid.Rows):
                term_num = int(data_grid.GetCell(row, 0))
                coeff = data_grid.GetCell(row, 1)  # 系数值 (波长单位)
                zernike_results['coefficients'][term_num] = coeff
            
            # 获取RMS和PV
            if data_grid.Rows > 0:
                zernike_results['rms_wavefront'] = data_grid.GetCell(0, 4) if data_grid.Columns > 4 else 0.0
                zernike_results['pv_wavefront'] = data_grid.GetCell(0, 5) if data_grid.Columns > 5 else 0.0
            
            return zernike_results
            
        except Exception as e:
            print(f"计算泽尼克系数失败：{e}")
            return None
    
    def calculate_image_quality_metrics(self, field_index: int = 1,
                                       wavelength_index: int = 1) -> Dict:
        """
        计算像质指标（RMS波前误差、Strehl Ratio等）
        
        Args:
            field_index: 视场点索引
            wavelength_index: 波长索引
            
        Returns:
            Dict: 像质指标字典
        """
        metrics = {
            'field_index': field_index,
            'wavelength_index': wavelength_index,
            'rms_wavefront': 0.0,
            'pv_wavefront': 0.0,
            'strehl_ratio': 0.0,
            'encircled_energy': {}
        }
        
        try:
            # 获取泽尼克系数来计算RMS和PV
            zernike_data = self.calculate_zernike_coefficients(field_index, wavelength_index)
            if zernike_data:
                metrics['rms_wavefront'] = zernike_data['rms_wavefront']
                metrics['pv_wavefront'] = zernike_data['pv_wavefront']
            
            # 计算Strehl Ratio (通过MTF分析或点列图)
            strehl_analysis = self.system.Analyses.New_Analysis_SettingsFirst("Standard Spot Diagram")
            strehl_settings = strehl_analysis.GetSettings()
            strehl_settings.FieldNumber = field_index
            strehl_settings.Wavelength = wavelength_index
            strehl_analysis.ApplyAndWaitForCompletion()
            
            # 近似Strehl Ratio (实际应从波前图获取)
            if metrics['rms_wavefront'] > 0:
                # Maréchal近似: Strehl ≈ exp(-(2π·RMS)²)
                rms_waves = metrics['rms_wavefront']
                metrics['strehl_ratio'] = np.exp(-(2 * np.pi * rms_waves) ** 2)
            
            return metrics
            
        except Exception as e:
            print(f"计算像质指标失败：{e}")
            return metrics
    
    def run_tolerance_analysis(self, 
                              surface_indices: List[int] = None,
                              error_ranges: Dict = None,
                              fields: List[int] = None,
                              num_samples: int = 5) -> Dict:
        """
        执行完整的公差分析
        
        Args:
            surface_indices: 要分析的表面列表，None则分析所有镜片表面
            error_ranges: 误差范围配置
                {
                    'decenter': (min, max),  # mm
                    'tilt': (min, max),      # 度
                    'thickness': (min, max), # mm
                    'refractive_index': (min, max),
                    'zernike_surface': {     # 泽尼克面型误差
                        'terms': [1, 2, 3, ...],  # 泽尼克项
                        'coeff_range': (min, max)  # μm
                    }
                }
            fields: 要分析的视场点列表，None则分析所有视场
            num_samples: 每个误差类型的采样点数
            
        Returns:
            Dict: 公差分析结果
        """
        if surface_indices is None:
            lens_surfaces = self.get_lens_surfaces()
            surface_indices = [s['index'] for s in lens_surfaces]
        
        if fields is None:
            fields = [f['index'] for f in self.fields]
        
        # 默认误差范围
        if error_ranges is None:
            error_ranges = {
                'decenter': (-0.05, 0.05),      # ±0.05 mm
                'tilt': (-0.5, 0.5),            # ±0.5 度
                'thickness': (-0.02, 0.02),     # ±0.02 mm
                'refractive_index': (-0.001, 0.001),
                'zernike_surface': {
                    'terms': list(range(1, 10)),  # Z1-Z9
                    'coeff_range': (-0.5, 0.5)    # ±0.5 μm
                }
            }
        
        results = {
            'surfaces_analyzed': surface_indices,
            'fields_analyzed': fields,
            'error_types': list(error_ranges.keys()),
            'data': []
        }
        
        print(f"\n开始公差分析:")
        print(f"  表面数量: {len(surface_indices)}")
        print(f"  视场点数量: {len(fields)}")
        print(f"  误差类型: {list(error_ranges.keys())}")
        print(f"  采样点数: {num_samples}")
        
        # 保存原始系统状态
        self.system.saveAs("tolerance_backup.zmx")
        
        sample_count = 0
        total_samples = len(surface_indices) * len(error_ranges) * num_samples
        
        # 遍历每个表面
        for surf_idx in surface_indices:
            print(f"\n分析表面 {surf_idx}:")
            surface_info = self.lens_data.GetSurfaceAt(surf_idx)
            print(f"  类型: {surface_info.Type}, 材料: {surface_info.Material}")
            
            # 遍历每种误差类型
            for error_type, error_range in error_ranges.items():
                if error_type == 'zernike_surface':
                    # 泽尼克面型误差特殊处理
                    zernike_config = error_range
                    terms = zernike_config['terms']
                    coeff_range = zernike_config['coeff_range']
                    
                    for sample in range(num_samples):
                        sample_count += 1
                        print(f"  [{sample_count}/{total_samples}] {error_type} (样本 {sample+1})")
                        
                        # 生成随机泽尼克系数
                        zernike_coeffs = {}
                        for term in terms:
                            coeff = np.random.uniform(coeff_range[0], coeff_range[1])
                            zernike_coeffs[term] = coeff
                        
                        # 重置系统
                        self.system.loadFile("tolerance_backup.zmx")
                        self._update_system_info()
                        
                        # 应用误差
                        self.apply_zernike_surface_error(surf_idx, zernike_coeffs)
                        
                        # 更新系统
                        self.system.LDE.Rebuild()
                        
                        # 计算所有视场的泽尼克系数
                        for field_idx in fields:
                            zernike_result = self.calculate_zernike_coefficients(field_idx)
                            metrics = self.calculate_image_quality_metrics(field_idx)
                            
                            if zernike_result and metrics:
                                result_entry = {
                                    'surface_index': surf_idx,
                                    'field_index': field_idx,
                                    'error_type': error_type,
                                    'sample': sample + 1,
                                    'zernike_coeffs': zernike_result['coefficients'],
                                    'rms_wavefront': metrics['rms_wavefront'],
                                    'pv_wavefront': metrics['pv_wavefront'],
                                    'strehl_ratio': metrics['strehl_ratio'],
                                    'applied_errors': zernike_coeffs
                                }
                                results['data'].append(result_entry)
                
                elif error_type == 'refractive_index':
                    # 折射率误差需要特殊处理
                    print(f"  跳过 {error_type} (需要材料数据库支持)")
                    continue
                
                else:
                    # 其他误差类型
                    for sample in range(num_samples):
                        sample_count += 1
                        print(f"  [{sample_count}/{total_samples}] {error_type} (样本 {sample+1})")
                        
                        # 生成随机误差值
                        error_value = np.random.uniform(error_range[0], error_range[1])
                        
                        # 重置系统
                        self.system.loadFile("tolerance_backup.zmx")
                        self._update_system_info()
                        
                        # 应用误差
                        if error_type == 'decenter':
                            decenter_x = error_value
                            decenter_y = np.random.uniform(error_range[0], error_range[1])
                            self.apply_decenter(surf_idx, decenter_x, decenter_y)
                        
                        elif error_type == 'tilt':
                            tilt_x = error_value
                            tilt_y = np.random.uniform(error_range[0], error_range[1])
                            tilt_z = np.random.uniform(error_range[0], error_range[1])
                            self.apply_tilt(surf_idx, tilt_x, tilt_y, tilt_z)
                        
                        elif error_type == 'thickness':
                            self.apply_thickness_error(surf_idx, error_value)
                        
                        # 更新系统
                        self.system.LDE.Rebuild()
                        
                        # 计算所有视场的泽尼克系数和像质指标
                        for field_idx in fields:
                            zernike_result = self.calculate_zernike_coefficients(field_idx)
                            metrics = self.calculate_image_quality_metrics(field_idx)
                            
                            if zernike_result and metrics:
                                result_entry = {
                                    'surface_index': surf_idx,
                                    'field_index': field_idx,
                                    'error_type': error_type,
                                    'sample': sample + 1,
                                    'zernike_coeffs': zernike_result['coefficients'],
                                    'rms_wavefront': metrics['rms_wavefront'],
                                    'pv_wavefront': metrics['pv_wavefront'],
                                    'strehl_ratio': metrics['strehl_ratio'],
                                    'applied_error_value': error_value
                                }
                                results['data'].append(result_entry)
        
        # 恢复原始系统
        self.system.loadFile("tolerance_backup.zmx")
        self._update_system_info()
        
        print(f"\n公差分析完成!")
        print(f"  总样本数: {len(results['data'])}")
        
        return results
    
    def export_results(self, results: Dict, output_file: str):
        """
        导出分析结果到CSV文件
        
        Args:
            results: 分析结果字典
            output_file: 输出文件路径
        """
        import csv
        
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # 写入表头
            header = ['Surface_Index', 'Field_Index', 'Error_Type', 'Sample', 
                     'RMS_Wavefront', 'PV_Wavefront', 'Strehl_Ratio']
            
            # 添加泽尼克系数列
            max_zernike_term = 0
            for entry in results['data']:
                if 'zernike_coeffs' in entry:
                    max_zernike_term = max(max_zernike_term, max(entry['zernike_coeffs'].keys(), default=0))
            
            for i in range(1, max_zernike_term + 1):
                header.append(f'Z{i}')
            
            header.append('Applied_Error_Details')
            writer.writerow(header)
            
            # 写入数据
            for entry in results['data']:
                row = [
                    entry['surface_index'],
                    entry['field_index'],
                    entry['error_type'],
                    entry['sample'],
                    entry['rms_wavefront'],
                    entry['pv_wavefront'],
                    entry['strehl_ratio']
                ]
                
                # 添加泽尼克系数
                zernike_coeffs = entry.get('zernike_coeffs', {})
                for i in range(1, max_zernike_term + 1):
                    row.append(zernike_coeffs.get(i, 0.0))
                
                # 添加应用的误差详情
                error_details = entry.get('applied_error_value', entry.get('applied_errors', ''))
                row.append(str(error_details))
                
                writer.writerow(row)
        
        print(f"结果已导出到: {output_file}")
    
    def close(self):
        """关闭OpticStudio连接"""
        if self.system:
            self.system.close()
        if self.zos:
            self.zos.disconnect()


# 使用示例
if __name__ == "__main__":
    # 初始化分析器
    analyzer = AdvancedToleranceAnalyzer()
    
    # 加载ZMX文件
    zmx_path = r"C:\path\to\your\lens.zmx"  # 修改为实际路径
    if not analyzer.load_system(zmx_path):
        exit(1)
    
    # 获取镜片表面
    lens_surfaces = analyzer.get_lens_surfaces()
    print(f"\n找到 {len(lens_surfaces)} 个镜片表面:")
    for surf in lens_surfaces:
        print(f"  表面 {surf['index']}: {surf['material']}")
    
    # 配置误差范围
    error_config = {
        'decenter': (-0.05, 0.05),      # ±0.05 mm
        'tilt': (-0.5, 0.5),            # ±0.5 度
        'thickness': (-0.02, 0.02),     # ±0.02 mm
        'zernike_surface': {
            'terms': list(range(1, 10)),  # Z1-Z9 (Fringe标准)
            'coeff_range': (-0.5, 0.5)    # ±0.5 μm
        }
    }
    
    # 执行公差分析
    results = analyzer.run_tolerance_analysis(
        surface_indices=[s['index'] for s in lens_surfaces],
        error_ranges=error_config,
        fields=None,  # 使用所有视场
        num_samples=3  # 每个误差类型3个样本
    )
    
    # 导出结果
    analyzer.export_results(results, "tolerance_analysis_results.csv")
    
    # 统计分析
    print("\n=== 公差分析统计 ===")
    all_rms = [d['rms_wavefront'] for d in results['data']]
    all_strehl = [d['strehl_ratio'] for d in results['data']]
    
    print(f"RMS波前误差: 均值={np.mean(all_rms):.4f}, 标准差={np.std(all_rms):.4f}")
    print(f"Strehl Ratio: 均值={np.mean(all_strehl):.4f}, 最小值={np.min(all_strehl):.4f}")
    
    # 关闭连接
    analyzer.close()

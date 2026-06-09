"""
ZMX 文件公差分析与泽尼克系数计算脚本

功能：
1. 读取指定路径的 ZMX 文件
2. 对每个镜片表面进行偏心 (Decenter) 和倾斜 (Tilt) 操作
3. 计算泽尼克系数 (Zernike coefficients) 等像质指标
4. 进行公差分析并输出结果

使用前请确保：
- 已安装 ZOSPy: pip install zospy
- 已安装 Ansys Zemax OpticStudio
- OpticStudio 处于 Interactive Extension 模式
"""

import numpy as np
from pathlib import Path
from typing import List, Dict, Any
import zospy as zp
from zospy.analyses.wavefront import ZernikeStandardCoefficients


class LensToleranceAnalyzer:
    """镜片公差分析器"""
    
    def __init__(self, zmx_path: str):
        """
        初始化分析器
        
        Parameters
        ----------
        zmx_path : str
            ZMX 文件的路径
        """
        self.zmx_path = Path(zmx_path)
        self.zos = None
        self.oss = None
        self.original_data = {}  # 存储原始镜片数据
        
    def connect(self):
        """连接到 OpticStudio"""
        print("正在连接到 OpticStudio...")
        self.zos = zp.ZOS()
        self.oss = self.zos.connect("extension")
        print("连接成功！")
        
    def load_system(self):
        """加载 ZMX 文件"""
        print(f"正在加载文件：{self.zmx_path}")
        self.oss.load_file(str(self.zmx_path))
        print("文件加载完成！")
        
    def get_lens_surfaces(self) -> List[int]:
        """
        获取所有镜片表面的索引
        
        Returns
        -------
        List[int]
            镜片表面索引列表
        """
        lens_surfaces = []
        lde = self.oss.LDE
        
        for i in range(lde.NumberOfSurfaces):
            surface = lde.GetSurfaceAt(i)
            # 跳过光阑面和像面，只处理有厚度和材料的镜片表面
            if surface.Thickness != 0 or (surface.Material and surface.Material.strip()):
                # 检查是否为标准光学表面
                if surface.Type.ToString() not in ['Stop', 'Image']:
                    lens_surfaces.append(i)
                    
        print(f"找到 {len(lens_surfaces)} 个镜片表面：{lens_surfaces}")
        return lens_surfaces
    
    def save_original_data(self, surface_indices: List[int]):
        """
        保存镜片的原始数据（用于恢复）
        
        Parameters
        ----------
        surface_indices : List[int]
            要保存的表面索引列表
        """
        print("保存原始镜片数据...")
        for idx in surface_indices:
            surface = self.oss.LDE.GetSurfaceAt(idx)
            # 使用 TiltDecenterData 来获取和设置偏心和倾斜数据
            tilt_decenter_data = surface.TiltDecenterData
            self.original_data[idx] = {
                'x_decenter': tilt_decenter_data.BeforeSurfaceDecenterX,
                'y_decenter': tilt_decenter_data.BeforeSurfaceDecenterY,
                'x_tilt': tilt_decenter_data.BeforeSurfaceTiltX,
                'y_tilt': tilt_decenter_data.BeforeSurfaceTiltY,
            }
        print("原始数据保存完成！")
        
    def apply_decenter_tilt(self, surface_index: int, 
                           x_decenter: float = 0.0, 
                           y_decenter: float = 0.0,
                           x_tilt: float = 0.0, 
                           y_tilt: float = 0.0):
        """
        对指定表面应用偏心和倾斜
        
        Parameters
        ----------
        surface_index : int
            表面索引
        x_decenter : float
            X 方向偏心量 (mm)
        y_decenter : float
            Y 方向偏心量 (mm)
        x_tilt : float
            X 方向倾斜角度 (度)
        y_tilt : float
            Y 方向倾斜角度 (度)
        """
        surface = self.oss.LDE.GetSurfaceAt(surface_index)
        # 使用 TiltDecenterData 来设置偏心和倾斜数据
        tilt_decenter_data = surface.TiltDecenterData
        tilt_decenter_data.BeforeSurfaceDecenterX = x_decenter
        tilt_decenter_data.BeforeSurfaceDecenterY = y_decenter
        tilt_decenter_data.BeforeSurfaceTiltX = x_tilt
        tilt_decenter_data.BeforeSurfaceTiltY = y_tilt
        
    def restore_original_data(self, surface_index: int):
        """
        恢复表面的原始数据
        
        Parameters
        ----------
        surface_index : int
            表面索引
        """
        if surface_index in self.original_data:
            data = self.original_data[surface_index]
            self.apply_decenter_tilt(
                surface_index,
                x_decenter=data['x_decenter'],
                y_decenter=data['y_decenter'],
                x_tilt=data['x_tilt'],
                y_tilt=data['y_tilt'],
            )
            
    def calculate_zernike_coefficients(self, 
                                       sampling: str = "64x64",
                                       maximum_term: int = 37,
                                       wavelength: int = 1,
                                       field: int = 1,
                                       surface: str = "Image") -> Dict[str, Any]:
        """
        计算泽尼克系数
        
        Parameters
        ----------
        sampling : str
            采样网格大小，如 "64x64"
        maximum_term : int
            最大项数
        wavelength : int
            波长编号
        field : int
            视场编号
        surface : str
            表面名称或索引
            
        Returns
        -------
        Dict[str, Any]
            泽尼克系数分析结果
        """
        zernike_analysis = ZernikeStandardCoefficients(
            sampling=sampling,
            maximum_term=maximum_term,
            wavelength=wavelength,
            field=field,
            surface=surface
        )
        
        result = zernike_analysis.run_analysis(self.oss)
        
        # 提取关键数据
        zernike_data = {
            'coefficients': result.coefficients,
            'rms_to_chief': result.from_integration_of_the_rays.rms_to_chief,
            'rms_to_centroid': result.from_integration_of_the_rays.rms_to_centroid,
            'strehl_ratio': result.from_integration_of_the_rays.strehl_ratio,
            'peak_to_valley_to_chief': result.peak_to_valley_to_chief,
            'peak_to_valley_to_centroid': result.peak_to_valley_to_centroid,
            'rms_fit_error': result.rms_fit_error,
        }
        
        return zernike_data
    
    def run_tolerance_analysis(self,
                              decenter_range: tuple = (-0.1, 0.1),
                              tilt_range: tuple = (-1.0, 1.0),
                              num_samples: int = 5,
                              output_file: str = "tolerance_analysis_results.txt"):
        """
        运行公差分析
        
        Parameters
        ----------
        decenter_range : tuple
            偏心范围 (min, max)，单位 mm
        tilt_range : tuple
            倾斜范围 (min, max)，单位 度
        num_samples : int
            每个参数的采样点数
        output_file : str
            输出文件名
        """
        print("\n" + "="*60)
        print("开始公差分析")
        print("="*60)
        
        # 获取所有镜片表面
        lens_surfaces = self.get_lens_surfaces()
        
        # 保存原始数据
        self.save_original_data(lens_surfaces)
        
        # 生成参数采样值
        decenter_values = np.linspace(decenter_range[0], decenter_range[1], num_samples)
        tilt_values = np.linspace(tilt_range[0], tilt_range[1], num_samples)
        
        results = []
        total_tests = len(lens_surfaces) * num_samples * num_samples
        
        test_count = 0
        for surface_idx in lens_surfaces:
            print(f"\n分析表面 {surface_idx}...")
            surface = self.oss.LDE.GetSurfaceAt(surface_idx)
            surface_name = f"Surface {surface_idx}"
            
            # 遍历偏心值
            for x_dec in decenter_values:
                for y_dec in decenter_values:
                    # 遍历倾斜值
                    for x_tilt in tilt_values:
                        for y_tilt in tilt_values:
                            test_count += 1
                            print(f"测试 {test_count}/{total_tests}: "
                                  f"表面={surface_idx}, X_dec={x_dec:.3f}, Y_dec={y_dec:.3f}, "
                                  f"X_tilt={x_tilt:.3f}, Y_tilt={y_tilt:.3f}")
                            
                            # 应用偏心和倾斜
                            self.apply_decenter_tilt(
                                surface_idx,
                                x_decenter=x_dec,
                                y_decenter=y_dec,
                                x_tilt=x_tilt,
                                y_tilt=y_tilt
                            )
                            
                            try:
                                # 计算泽尼克系数
                                zernike_result = self.calculate_zernike_coefficients()
                                
                                # 记录结果
                                result_entry = {
                                    'surface': surface_idx,
                                    'x_decenter': x_dec,
                                    'y_decenter': y_dec,
                                    'x_tilt': x_tilt,
                                    'y_tilt': y_tilt,
                                    'zernike_coefficients': zernike_result['coefficients'],
                                    'rms_wavefront': zernike_result['rms_to_chief'].value,
                                    'strehl_ratio': zernike_result['strehl_ratio'],
                                    'pv_wavefront': zernike_result['peak_to_valley_to_chief'].value,
                                }
                                results.append(result_entry)
                                
                            except Exception as e:
                                print(f"  分析失败：{e}")
                                
                            # 恢复原始状态
                            self.restore_original_data(surface_idx)
        
        # 保存结果到文件
        self._save_results(results, output_file)
        
        # 打印统计摘要
        self._print_summary(results)
        
        return results
    
    def _save_results(self, results: List[Dict], filename: str):
        """保存结果到文件"""
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("公差分析报告 - 泽尼克系数分析\n")
            f.write("="*80 + "\n\n")
            
            for i, result in enumerate(results):
                f.write(f"测试 #{i+1}\n")
                f.write("-"*40 + "\n")
                f.write(f"表面索引：{result['surface']}\n")
                f.write(f"X 偏心：{result['x_decenter']:.4f} mm\n")
                f.write(f"Y 偏心：{result['y_decenter']:.4f} mm\n")
                f.write(f"X 倾斜：{result['x_tilt']:.4f} deg\n")
                f.write(f"Y 倾斜：{result['y_tilt']:.4f} deg\n")
                f.write(f"RMS 波前误差：{result['rms_wavefront']:.6f} waves\n")
                f.write(f"Strehl Ratio: {result['strehl_ratio']:.6f}\n")
                f.write(f"PV 波前误差：{result['pv_wavefront']:.6f} waves\n")
                
                f.write("\n泽尼克系数:\n")
                for term_idx, coeff in result['zernike_coefficients'].items():
                    f.write(f"  Z{term_idx}: {coeff.value:.6e} ({coeff.formula})\n")
                
                f.write("\n")
        
        print(f"\n结果已保存到：{filename}")
        
    def _print_summary(self,results: List[Dict]):
        """打印统计摘要"""
        print("\n" + "="*60)
        print("公差分析摘要")
        print("="*60)
        
        if not results:
            print("无有效结果")
            return
            
        rms_values = [r['rms_wavefront'] for r in results]
        strehl_values = [r['strehl_ratio'] for r in results]
        pv_values = [r['pv_wavefront'] for r in results]
        
        print(f"\n总测试次数：{len(results)}")
        print(f"\nRMS 波前误差:")
        print(f"  最小值：{min(rms_values):.6f} waves")
        print(f"  最大值：{max(rms_values):.6f} waves")
        print(f"  平均值：{np.mean(rms_values):.6f} waves")
        print(f"  标准差：{np.std(rms_values):.6f} waves")
        
        print(f"\nStrehl Ratio:")
        print(f"  最小值：{min(strehl_values):.6f}")
        print(f"  最大值：{max(strehl_values):.6f}")
        print(f"  平均值：{np.mean(strehl_values):.6f}")
        
        print(f"\nPV 波前误差:")
        print(f"  最小值：{min(pv_values):.6f} waves")
        print(f"  最大值：{max(pv_values):.6f} waves")
        print(f"  平均值：{np.mean(pv_values):.6f} waves")
        
        # 找出最差情况
        worst_idx = np.argmax(rms_values)
        print(f"\n最差情况 (最大 RMS):")
        print(f"  表面：{results[worst_idx]['surface']}")
        print(f"  X 偏心：{results[worst_idx]['x_decenter']:.4f} mm")
        print(f"  Y 偏心：{results[worst_idx]['y_decenter']:.4f} mm")
        print(f"  X 倾斜：{results[worst_idx]['x_tilt']:.4f} deg")
        print(f"  Y 倾斜：{results[worst_idx]['y_tilt']:.4f} deg")
        print(f"  RMS: {results[worst_idx]['rms_wavefront']:.6f} waves")
        print(f"  Strehl: {results[worst_idx]['strehl_ratio']:.6f}")
        
    def close(self):
        """关闭连接"""
        if self.oss:
            print("\n关闭 OpticStudio 连接...")
            self.oss.close()
            print("连接已关闭")


def main():
    """主函数示例"""
    # 替换为你的 ZMX 文件路径
    zmx_file = "/workspace/examples/Retinal illumination in pseudophakic eyes with and without Negative Dysphotopsia/PseudophakicControlModel.zmx"
    
    # 创建分析器
    analyzer = LensToleranceAnalyzer(zmx_file)
    
    try:
        # 连接并加载系统
        analyzer.connect()
        analyzer.load_system()
        
        # 运行公差分析
        # 参数说明：
        # - decenter_range: 偏心范围 (mm)
        # - tilt_range: 倾斜范围 (度)
        # - num_samples: 每个参数的采样点数 (增加此值会显著增加计算时间)
        results = analyzer.run_tolerance_analysis(
            decenter_range=(-0.05, 0.05),  # ±0.05mm 偏心
            tilt_range=(-0.5, 0.5),         # ±0.5 度倾斜
            num_samples=3,                  # 每个参数 3 个点
            output_file="tolerance_results.txt"
        )
        
        print("\n分析完成!")
        
    except Exception as e:
        print(f"发生错误：{e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # 关闭连接
        analyzer.close()


if __name__ == "__main__":
    main()

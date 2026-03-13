import matplotlib.font_manager as fm
fonts = [f.name for f in fm.fontManager.ttflist if 'Hei' in f.name or 'YaHei' in f.name or 'Sim' in f.name]
print("可用中文字体:", set(fonts))
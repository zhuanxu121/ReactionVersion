import tkinter as tk
from tkinter import messagebox
from tkinter import font as tkfont
import pandas as pd
import os

# ================ RDKit 相关 ================
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from PIL import Image, ImageTk

class ReactionChainViewer:
    def __init__(self, master):
        self.master = master
        master.title("反应方程式查看器")
        master.geometry("1500x700")  # 主窗口大小

        # ---- 一些可调参数 ----
        self.scale_factor = 50.0    # px/Å，控制整体原子大小一致
        self.min_size = (80, 80)    # 分子图最小尺寸
        self.max_size = (300, 300)  # 分子图最大尺寸

        self.plus_font_size = 20    # "+" 符号字体大小
        self.arrow_font_size = 20   # "⟹" 符号字体大小

        self.h_margin = 20          # 左右边缘留白
        self.v_margin = 20          # 上下边缘留白
        self.item_gap = 10          # 行内元素之间的水平间距
        self.line_gap = 20          # 行与行之间的垂直间距

        # 用于保存 PhotoImage 引用，避免被垃圾回收
        self.image_refs = []

        # 默认 CSV 文件路径
        self.default_path = "../Dataset/47083204_Trans_G2S_val.csv"
        self.file_path = self.default_path
        self.chain_index = 0
        self.chains = []

        # ------------------- Canvas + 滚动条 -------------------
        self.display_frame = tk.Frame(master, bg="white")
        self.display_frame.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(self.display_frame, bg="white")
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # 垂直滚动条
        self.v_scroll = tk.Scrollbar(self.display_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        self.v_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.configure(yscrollcommand=self.v_scroll.set)

        # 绑定鼠标滚轮事件（Windows 与 Linux 事件名不同）
        self.canvas.bind_all("<MouseWheel>", self.on_mousewheel)       # Windows
        self.canvas.bind_all("<Button-4>", self.on_mousewheel)         # Linux: 滚轮向上
        self.canvas.bind_all("<Button-5>", self.on_mousewheel)         # Linux: 滚轮向下

        # ------------------ 控制面板 ------------------
        self.control_panel = tk.Frame(master)
        self.control_panel.place(relx=1.0, rely=1.0, anchor="se")

        # 第一行：文件路径设置与前后切换
        self.gear_button = tk.Button(
            self.control_panel,
            text="⚙",
            command=self.open_file_path_window,
            font=("Arial", 14)
        )
        self.gear_button.grid(row=0, column=0, padx=5, pady=5)

        self.left_button = tk.Button(
            self.control_panel,
            text="←",
            command=self.show_previous_chain,
            font=("Arial", 14)
        )
        self.left_button.grid(row=0, column=1, padx=5, pady=5)

        self.right_button = tk.Button(
            self.control_panel,
            text="→",
            command=self.show_next_chain,
            font=("Arial", 14)
        )
        self.right_button.grid(row=0, column=2, padx=5, pady=5)

        # 第二行：显示当前数据编号与总数据数，并提供跳转入口
        self.index_frame = tk.Frame(self.control_panel)
        self.index_frame.grid(row=1, column=0, columnspan=3, pady=(10,5))

        self.current_index_label = tk.Label(self.index_frame, text="当前数据：", font=("Arial", 12))
        self.current_index_label.grid(row=0, column=0, padx=5)

        self.index_entry = tk.Entry(self.index_frame, width=5, font=("Arial", 12))
        self.index_entry.grid(row=0, column=1, padx=5)
        self.index_entry.insert(0, "1")  # 初始显示第一条数据

        self.total_label = tk.Label(self.index_frame, text="/ 0", font=("Arial", 12))
        self.total_label.grid(row=0, column=2, padx=5)

        self.go_button = tk.Button(self.index_frame, text="前往", font=("Arial", 12), command=self.go_to_index)
        self.go_button.grid(row=0, column=3, padx=5)

        # 初次加载数据并显示第一个反应链
        self.load_data(self.file_path)
        self.master.after(100, self.display_chain)  # 等待窗口初始化完再渲染

    # ================== 鼠标滚轮事件 ===================
    def on_mousewheel(self, event):
        """根据不同系统的事件，控制 Canvas 上下滚动"""
        if event.num == 5 or event.delta < 0:
            self.canvas.yview_scroll(1, "units")
        elif event.num == 4 or event.delta > 0:
            self.canvas.yview_scroll(-1, "units")

    # ================== 文件路径设置 ===================
    def open_file_path_window(self):
        """弹出窗口用于输入 CSV 文件路径"""
        self.new_window = tk.Toplevel(self.master)
        self.new_window.title("设置CSV文件路径")
        self.new_window.geometry("400x100")

        tk.Label(self.new_window, text="请输入CSV文件路径:", font=("Arial", 12)).pack(pady=5)
        self.path_entry = tk.Entry(self.new_window, width=50)
        self.path_entry.pack(pady=5)
        self.path_entry.insert(0, self.file_path)

        tk.Button(self.new_window, text="确定", command=self.set_file_path, font=("Arial", 12)).pack(pady=5)

    def set_file_path(self):
        new_path = self.path_entry.get()
        if not os.path.exists(new_path):
            messagebox.showerror("错误", "文件路径不存在!")
        else:
            self.file_path = new_path
            self.chain_index = 0  # 重置索引
            self.load_data(self.file_path)
            self.display_chain()
            self.new_window.destroy()

    # ================== 数据加载与分组 ===================
    def load_data(self, path):
        """读取 CSV 文件，并按照反应链规则对数据进行分组"""
        try:
            df = pd.read_csv(path)
            self.chains = self.group_chains(df)
        except Exception as e:
            messagebox.showerror("错误", f"读取文件出错: {e}")
            self.chains = []

    def group_chains(self, df):
        """
        将 CSV 中每一行的反应方程式分组：
        按行读取，当遇到左右两边相同的反应（代表“产物 >> 产物”）时认为该链结束
        """
        chains = []
        current_chain = []
        for _, row in df.iterrows():
            rxn = row['rxn_smiles']
            left_mols, right_mols = self.parse_reaction(rxn)
            current_chain.append((left_mols, right_mols))
            if "+".join(left_mols) == "+".join(right_mols):
                chains.append(current_chain)
                current_chain = []
        if current_chain:
            chains.append(current_chain)
        return chains

    def parse_reaction(self, rxn_smiles):
        """
        将一个反应式（SMILES格式）转换为分子列表：
         – 按 ">>" 分割为左右两部分
         – 左右部分中的多个分子以 "." 分隔
        """
        try:
            left, right = rxn_smiles.split(">>")
        except ValueError:
            left, right = rxn_smiles, ""
        left_mols = left.split(".")
        right_mols = right.split(".")
        return left_mols, right_mols

    # ================== 分子图生成逻辑 ===================
    def smiles_to_tk_image(self, smiles):
        """
        使用 RDKit 绘制分子图像，并转换为 Tkinter 的 PhotoImage。
        根据分子边界和全局 scale_factor 计算图像尺寸，保证“原子大小”一致。
        """
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return None, 0, 0

        AllChem.Compute2DCoords(mol)
        conf = mol.GetConformer()
        xs = []
        ys = []
        for i in range(mol.GetNumAtoms()):
            pos = conf.GetAtomPosition(i)
            xs.append(pos.x)
            ys.append(pos.y)
        if not xs or not ys:
            return None, 0, 0

        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        width_angs = max_x - min_x
        height_angs = max_y - min_y

        raw_w = int(width_angs * self.scale_factor + 0.5)
        raw_h = int(height_angs * self.scale_factor + 0.5)
        w = max(self.min_size[0], min(raw_w, self.max_size[0]))
        h = max(self.min_size[1], min(raw_h, self.max_size[1]))

        img = Draw.MolToImage(mol, size=(w, h))
        tk_img = ImageTk.PhotoImage(img)
        return tk_img, w, h

    # ================== 构造“可视元素”列表 ===================
    def build_item_list(self, chain):
        """
        将一个反应链（如 [ (left_mols, right_mols), ... ]）
        转换为一个按顺序布局的“元素列表”。
        每个元素是一个 dict：
            {
                "type": "image"/"text",
                "width": ...,
                "height": ...,
                "data": ... (PhotoImage 或字符串),
                "font": ... (若是 text 才有)
            }
        """
        items = []
        font_plus = tkfont.Font(family="Arial", size=self.plus_font_size)
        font_arrow = tkfont.Font(family="Arial", size=self.arrow_font_size)

        step_count = len(chain)
        for i, (left_mols, right_mols) in enumerate(chain):
            for j, smi in enumerate(left_mols):
                tk_img, w, h = self.smiles_to_tk_image(smi)
                if tk_img:
                    items.append({"type": "image", "width": w, "height": h, "data": tk_img})
                if j < len(left_mols) - 1:
                    plus_text = "+"
                    plus_w = font_plus.measure(plus_text)
                    plus_h = font_plus.metrics("ascent") + font_plus.metrics("descent")
                    items.append({"type": "text", "width": plus_w, "height": plus_h, "data": plus_text, "font": font_plus})
            arrow_text = "⟹"
            arrow_w = font_arrow.measure(arrow_text)
            arrow_h = font_arrow.metrics("ascent") + font_arrow.metrics("descent")
            items.append({"type": "text", "width": arrow_w, "height": arrow_h, "data": arrow_text, "font": font_arrow})
            for k, smi in enumerate(right_mols):
                tk_img, w, h = self.smiles_to_tk_image(smi)
                if tk_img:
                    items.append({"type": "image", "width": w, "height": h, "data": tk_img})
                if k < len(right_mols) - 1:
                    plus_text = "+"
                    plus_w = font_plus.measure(plus_text)
                    plus_h = font_plus.metrics("ascent") + font_plus.metrics("descent")
                    items.append({"type": "text", "width": plus_w, "height": plus_h, "data": plus_text, "font": font_plus})
            if i < step_count - 1:
                sep_text = "⟹"
                sep_w = font_arrow.measure(sep_text)
                sep_h = font_arrow.metrics("ascent") + font_arrow.metrics("descent")
                items.append({"type": "text", "width": sep_w, "height": sep_h, "data": sep_text, "font": font_arrow})
        return items

    # ================== 行级布局并渲染 ===================
    def layout_and_render_items_on_canvas(self, items):
        """
        将“可视元素”列表 items 按行排布并在 Canvas 上绘制：
         - 同一行中，元素均以行最大高度为基准垂直居中对齐；
         - 若剩余宽度不足，则换行。
        """
        self.canvas.delete("all")
        self.image_refs.clear()

        canvas_width = self.canvas.winfo_width()
        x_offset = self.h_margin
        y_offset = self.v_margin

        row_items = []
        row_max_height = 0

        def flush_row():
            nonlocal y_offset, row_items, row_max_height
            center_y = y_offset + row_max_height / 2
            for (elem, x_start) in row_items:
                elem_center_x = x_start + elem["width"] / 2
                if elem["type"] == "image":
                    self.canvas.create_image(elem_center_x, center_y, anchor="center", image=elem["data"])
                    self.image_refs.append(elem["data"])
                else:
                    self.canvas.create_text(elem_center_x, center_y, anchor="center", text=elem["data"], font=elem["font"])
            y_offset += row_max_height + self.line_gap
            row_items.clear()
            row_max_height = 0

        for elem in items:
            if x_offset + elem["width"] + self.h_margin > canvas_width:
                flush_row()
                x_offset = self.h_margin
            row_items.append((elem, x_offset))
            row_max_height = max(row_max_height, elem["height"])
            x_offset += elem["width"] + self.item_gap

        if row_items:
            flush_row()

        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    # ================== 更新数据编号显示 ===================
    def update_index_info(self):
        """更新当前显示数据编号与总数据数"""
        total = len(self.chains)
        self.total_label.config(text=f"/ {total}")
        self.index_entry.delete(0, tk.END)
        self.index_entry.insert(0, str(self.chain_index + 1))

    # ================== 主显示逻辑 ===================
    def display_chain(self):
        """在 Canvas 上显示当前反应链，并更新编号显示"""
        if not self.chains or not (0 <= self.chain_index < len(self.chains)):
            self.canvas.delete("all")
            self.canvas.create_text(50, 50, anchor="nw", text="没有反应方程式可显示", font=("Arial", 16))
            self.update_index_info()
            return

        chain = self.chains[self.chain_index]
        items = self.build_item_list(chain)
        self.layout_and_render_items_on_canvas(items)
        self.update_index_info()

    # ================== 翻页按钮 ===================
    def show_previous_chain(self):
        """显示前一个反应链"""
        if self.chain_index > 0:
            self.chain_index -= 1
            self.display_chain()

    def show_next_chain(self):
        """显示下一个反应链"""
        if self.chain_index < len(self.chains) - 1:
            self.chain_index += 1
            self.display_chain()

    # ================== 跳转到指定数据 ===================
    def go_to_index(self):
        """读取 Entry 中的编号，跳转到对应数据"""
        try:
            idx = int(self.index_entry.get()) - 1
            if 0 <= idx < len(self.chains):
                self.chain_index = idx
                self.display_chain()
            else:
                messagebox.showerror("错误", "数据编号超出范围！")
        except ValueError:
            messagebox.showerror("错误", "请输入有效的数字！")

def run_viewer():
    """
    封装启动程序的逻辑，
    这样你可以在其他地方导入这个模块后，调用 run_viewer() 来启动 GUI，
    而不会在导入时自动运行。
    """
    root = tk.Tk()
    app = ReactionChainViewer(root)
    root.mainloop()

# 如果你直接运行这个文件，则调用 run_viewer() 启动程序
if __name__ == "__main__":
    run_viewer()

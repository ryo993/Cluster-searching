import numpy as np
from ase import Atoms
from ase.io import Trajectory, write
from ase.optimize import BFGS
from ase.optimize.basin import BasinHopping # 元のクラス
from chgnet.model.dynamics import CHGNetCalculator
from ase.neighborlist import neighbor_list # 距離チェック用

# =========================================================
# CHGNetモデルの読み込み (省略)
# =========================================================
print("CHGNetモデルを読み込んでいます...")
calc = CHGNetCalculator()

# =========================================================
# 初期構造の生成 (原子数に伴いradiusを変更してください。)
# =========================================================
def create_random_cluster(n_Ti=4, n_O=8, radius=3.0):
    symbols = ['Ti'] * n_Ti + ['O'] * n_O
    positions = []
    
    for _ in range(len(symbols)):
        while True:
            r = radius * np.random.rand()**(1/3)
            theta = np.arccos(2 * np.random.rand() - 1)
            phi = 2 * np.pi * np.random.rand()
            
            x = r * np.sin(theta) * np.cos(phi)
            y = r * np.sin(theta) * np.sin(phi)
            z = r * np.cos(theta)
            pos = np.array([x, y, z])
            
            if len(positions) == 0:
                positions.append(pos)
                break
            else:
                dists = np.linalg.norm(np.array(positions) - pos, axis=1)
                if np.all(dists > 1.0):
                    positions.append(pos)
                    break
                    
    atoms = Atoms(symbols, positions=positions)
    atoms.set_cell([15.0, 15.0, 15.0])
    atoms.center()
    atoms.set_pbc(False) 
    return atoms

print("TiO2クラスターの初期構造を作成しています...")
cluster = create_random_cluster()
cluster.calc = calc 

# =========================================================
# 最大500歩で強制打ち切りするカスタムBFGS (省略)
# =========================================================
class LimitedBFGS(BFGS):
    def run(self, fmax=None, steps=None):
        max_allowed_steps = 500
        if steps is None:
            steps = max_allowed_steps
        else:
            steps = min(steps, max_allowed_steps)
        return super().run(fmax=fmax, steps=steps)

# =========================================================
# ジャンプ後に原子間距離のチェックを行うカスタムBasinHopping
# =========================================================
class SafeBasinHopping(BasinHopping):
    def _check_overlap(self, atoms):
        # 重なりと判定する閾値（単位：Å）
        # C-Hの結合距離は約1.1Åなので、1.0Å未満を警告対象とします
        min_allowed_dist = 1.0 
        
        # セルが巨大なので、セルの壁を越えた干渉は気にしなくて良い
        i, j, d = neighbor_list('ijd', atoms, cutoff=min_allowed_dist)
        # 自身とのペア(i==j)を除外し、他の原子との距離だけチェック
        mask = i != j
        
        if len(d[mask]) > 0:
            actual_min = np.min(d[mask])
            print(f"  [Warning] 重なり検知 (dist={actual_min:.2f}Å < {min_allowed_dist}Å)。ジャンプをやり直します。")
            return True # 重なりあり
        return False # 重なりなし

    def random_swap(self):
        # 本物のジャンプ（random_swap）メソッドをオーバーライドします
        attempts = 0
        max_attempts = 100 # 100回まで再トライ
        
        while attempts < max_attempts:
            # ジャンプ前の位置を記憶
            old_positions = self.atoms.get_positions().copy()
            
            # 元のクラスのジャンプ処理を実行
            super().random_swap()
            
            # ジャンプ後の構造で重なりチェック
            if not self._check_overlap(self.atoms):
                break # 重なりがなければ終了（ジャンプ成功）
            
            # 重なりがあったら元に戻してやり直し
            self.atoms.set_positions(old_positions)
            attempts += 1
        
        if attempts == max_attempts:
            print("  [Warning] 100回試行しましたが重なりを回避できませんでした。そのまま進めます。")

# =========================================================
# 4. Basin Hopping の設定と実行
# =========================================================

# 通常の「BasinHopping」の代わりに、カスタムした「SafeBasinHopping」を使用！
bh = SafeBasinHopping(
    atoms=cluster,
    temperature=0.1,  
    dr=0.5,           
    trajectory='bh_optimization_Safe1000.traj',
    optimizer=LimitedBFGS, 
    local_minima_trajectory='local_minima_Safe1000.traj',
    fmax=0.2  
)

n_steps = 1000
print(f"1000ステップの『原子重なり防止付き』構造探索を開始します...")

# nohup python search_cluster.py > output.log &
bh.run(n_steps) 

print("探索が完了しました！")
# train_min_balanced.py  —  compact & balanced baseline

import os, sys, argparse, random
from pathlib import Path
import numpy as np
from PIL import ImageFile; ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix

# ---------------- basics ----------------
def seed_all(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

def device():
    if torch.cuda.is_available(): return torch.device("cuda")
    if hasattr(torch.backends,"mps") and torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")

def find_data_dir(user_arg: str|None) -> Path:
    if user_arg:
        p = Path(user_arg).expanduser().resolve()
        if (p/"train").is_dir() and (p/"val").is_dir() and (p/"test").is_dir(): return p
        sys.exit(f"[ERR] invalid --data: {p}")
    here = Path(__file__).resolve()
    for cand in [Path.cwd()/ "data", here.parent/ "data", here.parent.parent/ "data"]:
        if (cand/"train").is_dir() and (cand/"val").is_dir() and (cand/"test").is_dir():
            return cand.resolve()
    sys.exit("[ERR] data/train|val|test 폴더를 찾지 못했습니다. --data로 경로를 지정하세요.")

# ------------- class-conditional transform -------------
class CCImageFolder(datasets.ImageFolder):
    def __init__(self, root, t_common, t_clean, clean_name="clean"):
        super().__init__(root, transform=None)
        self.t_common, self.t_clean = t_common, t_clean
        if clean_name not in self.class_to_idx:
            raise ValueError(f"'{clean_name}' not in {list(self.class_to_idx)}")
        self.clean_idx = self.class_to_idx[clean_name]
    def __getitem__(self, i):
        path, y = self.samples[i]
        img = self.loader(path)
        x = self.t_clean(img) if y==self.clean_idx else self.t_common(img)
        return x, y

# ---------------- pipeline ----------------
def build_transforms(size=224):
    t_val = transforms.Compose([
        transforms.Resize((size,size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    t_common = transforms.Compose([
        transforms.Resize((size,size)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(0.2,0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    t_clean = transforms.Compose([
        transforms.Resize((size,size)),
        transforms.RandomResizedCrop(size, scale=(0.6,1.0), ratio=(0.8,1.25)),
        transforms.RandomHorizontalFlip(0.8),
        transforms.RandomRotation(25),
        transforms.ColorJitter(0.4,0.4,0.3,0.05),
        transforms.RandomApply([transforms.GaussianBlur(3)], p=0.3),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    return t_common, t_clean, t_val

def build_loaders(data_dir: Path, batch=16, size=224):
    t_common, t_clean, t_val = build_transforms(size)
    tr = CCImageFolder(data_dir/"train", t_common, t_clean, clean_name="clean")
    va = datasets.ImageFolder(data_dir/"val",  transform=t_val)
    te = datasets.ImageFolder(data_dir/"test", transform=t_val)

    # 클래스 불균형 보정: WeightedRandomSampler (파일 복제 없이 oversampling 효과)
    labels = [t for _, t in tr.samples]
    counts = np.bincount(labels, minlength=len(tr.classes))
    w_per_class = 1.0/np.clip(counts,1,None)
    w_samples = [w_per_class[t] for t in labels]
    sampler = WeightedRandomSampler(w_samples, num_samples=len(w_samples), replacement=True)

    train_ld = DataLoader(tr, batch_size=batch, sampler=sampler, num_workers=2)
    val_ld   = DataLoader(va, batch_size=batch, shuffle=False, num_workers=2)
    test_ld  = DataLoader(te, batch_size=batch, shuffle=False, num_workers=2)
    return tr.classes, counts, train_ld, val_ld, test_ld

def build_model(nc, dev):
    m = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
    m.classifier[3] = nn.Linear(m.classifier[3].in_features, nc)
    return m.to(dev)

# ---------------- train/eval ----------------
def run_epoch(model, loader, crit, opt, dev, train=True):
    model.train() if train else model.eval()
    totl=totc=totn=0
    with torch.set_grad_enabled(train):
        for x,y in loader:
            x,y = x.to(dev), y.to(dev)
            out = model(x); loss = crit(out,y)
            if train:
                opt.zero_grad(); loss.backward(); opt.step()
            totl += loss.item()*x.size(0)
            totc += (out.argmax(1)==y).sum().item()
            totn += x.size(0)
    return totl/max(1,totn), totc/max(1,totn)

def eval_report(model, loader, dev, names):
    model.eval(); P,L=[],[]
    with torch.no_grad():
        for x,y in loader:
            p = model(x.to(dev)).argmax(1).cpu().numpy()
            P.extend(p); L.extend(y.numpy())
    print("\n=== Classification Report ===")
    print(classification_report(L, P, target_names=names, digits=4))
    print("Confusion Matrix (rows=Actual, cols=Pred):")
    print(confusion_matrix(L, P))

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default=None)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--save", type=str, default="checkpoints/mobilenet_best.pth")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    seed_all(args.seed)
    dev = device()
    data_dir = find_data_dir(args.data)

    names, counts, train_ld, val_ld, test_ld = build_loaders(data_dir, batch=args.batch, size=args.img_size)
    print(f"[Device] {dev} | [Classes] {names} | [Counts train] {counts.tolist()}")

    model = build_model(len(names), dev)

    # 클래스 가중치(적은 클래스에 더 큰 가중) — clean recall 개선
    counts = np.array(counts, dtype=np.float32)
    w = (counts.sum() / np.clip(counts,1,None)).astype(np.float32)
    crit = nn.CrossEntropyLoss(weight=torch.tensor(w, device=dev))
    opt = optim.Adam(model.parameters(), lr=3e-4)

    best, save_path = -1.0, Path(args.save); save_path.parent.mkdir(parents=True, exist_ok=True)
    for ep in range(1, args.epochs+1):
        tr_l, tr_a = run_epoch(model, train_ld, crit, opt, dev, train=True)
        va_l, va_a = run_epoch(model, val_ld,   crit, opt, dev, train=False)
        print(f"[{ep:02d}] Train {tr_a:.3f}/{tr_l:.4f} | Val {va_a:.3f}/{va_l:.4f}")
        if va_a > best:
            best = va_a; torch.save(model.state_dict(), save_path)
            print(f"  ↳ saved best: {save_path} (val_acc={best:.4f})")

    model.load_state_dict(torch.load(save_path, map_location=dev))
    print(f"[Eval] loaded best: {save_path}")
    eval_report(model, test_ld, dev, names)

if __name__ == "__main__":
    main()

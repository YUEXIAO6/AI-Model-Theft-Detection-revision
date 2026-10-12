
import pandas as pd

df = pd.read_csv("traffic_logs.csv")
df["is_attack"] = df["label"].astype(str).str.lower().eq("attack")
credit = df["endpoint"].eq("/api/v1/credit_score")

n_all, n_atk = len(df), int(df["is_attack"].sum())
n_cr = int(credit.sum()); n_cr_atk = int((credit & df["is_attack"]).sum())
print(f" {n_all}")
print(f"label=Attack   : {n_atk}  ->  {n_atk/n_all*100:.2f}%")
print(f"credit_score          : {n_cr}  ({n_cr/n_all*100:.1f}% of all)")
print(f"credit : {n_cr_atk}  -> 占 credit 请求 {n_cr_atk/max(n_cr,1)*100:.2f}%")
print(f"credit: {n_cr_atk/n_all*100:.2f}%")


g = df.groupby("type").agg(total=("is_attack", "size"), attack=("is_attack", "sum"))
g["credit"] = df[credit].groupby("type").size()
g["attack_credit"] = df[credit & df["is_attack"]].groupby("type").size()
print(g.fillna(0).astype(int).to_string())


atk_ent = df[df["api_key"].astype(str).str.startswith("atk")]
print(f" {len(atk_ent)}， {len(atk_ent)/n_all*100:.2f}%")
print(atk_ent.groupby(atk_ent["api_key"].str.extract(r'(atk_s\d)')[0])
      .apply(lambda x: pd.Series({
          "requests": len(x),
          "credit_attack": int((x["endpoint"].eq('/api/v1/credit_score') & x["is_attack"]).sum())
      })).to_string())
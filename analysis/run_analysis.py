import jsonlines
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def selected_tokens_stats():
    df = pd.DataFrame({})
    with jsonlines.open("analysis/output.json", mode="r") as reader:
        for dict in reader:
            tmp_df = pd.DataFrame(dict)
            df_unique = tmp_df.groupby(["targets", "decoded_tokens"]).agg(
                excess_loss_mean=("excess_loss", "mean"),
                count=("excess_loss", "size")
            ).reset_index()
            df_unique.columns = ["token_id", "token", "excess_loss_mean", "count"]
            ignore_count_df = (
                tmp_df.loc[tmp_df["selected_targets"] == -100, ["targets"]]
                .value_counts()
                .reset_index(name="ignore_count")
                .rename(columns={"targets": "token_id"})
            )
            df_unique = df_unique.merge(ignore_count_df, on="token_id", how="left").fillna({"ignore_count": 0})
            df_unique["ignore_count"] = df_unique["ignore_count"].astype(int)
            
            df = pd.concat([df, df_unique])
    
    df = df.groupby(["token_id", "token"]).agg(
        excess_loss_mean=("excess_loss_mean", "mean"),
        count=("count", "sum"),
        ignore_count=("ignore_count", "sum")
    ).reset_index()
    df["selected/ignore"] = (df["count"] - df["ignore_count"]) / df["count"] * 100
    df["selection_ratio"] = df["selected/ignore"].map("{:.2f}%".format)
    df = df.sort_values(by=["count"], ascending=False).reset_index()
    print(df)
    df = df.sort_values(by=["excess_loss_mean"], ascending=False).reset_index()
    print(df)


if __name__ == "__main__":
    selected_tokens_stats()
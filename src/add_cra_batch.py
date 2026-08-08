from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "sred_training_data.csv"
EXPECTED_COLUMNS = ["text", "label"]
VALID_LABELS = {"routine", "borderline", "needs_more_info", "strong_sred"}


CRA_BATCH = [
    {
        "text": "Existing blockchain consensus mechanisms could not maintain transaction validation speed while preserving network integrity under high transaction volume, so the team tested multiple validation approaches and compared failure rates, processing speed, and consistency across iterations",
        "label": "strong_sred",
    },
    {
        "text": "The company used an existing blockchain API to record transactions and display wallet balances in a dashboard",
        "label": "routine",
    },
    {
        "text": "The team developed a blockchain-based platform",
        "label": "needs_more_info",
    },
    {
        "text": "The team compared available blockchain frameworks but did not explain what technological uncertainty existed or what new technical knowledge was gained",
        "label": "borderline",
    },
    {
        "text": "Known image recognition models failed to classify plant disease symptoms under variable lighting and overlapping leaf damage, so the team tested preprocessing methods and model architectures to improve detection reliability",
        "label": "strong_sred",
    },
    {
        "text": "The company used an existing image recognition service to identify plant images uploaded by users",
        "label": "routine",
    },
    {
        "text": "The team improved crop disease detection using artificial intelligence",
        "label": "needs_more_info",
    },
    {
        "text": "The team tested several plant image filters to improve classification but the description does not explain whether standard methods were insufficient or what was learned",
        "label": "borderline",
    },
    {
        "text": "Standard data reconciliation methods failed when transaction records arrived out of order from multiple systems, so the team tested custom matching logic and measured duplicate rates, mismatch rates, and processing reliability",
        "label": "strong_sred",
    },
    {
        "text": "The company imported transaction records into accounting software using standard CSV upload tools",
        "label": "routine",
    },
    {
        "text": "We created a new transaction processing system",
        "label": "needs_more_info",
    },
    {
        "text": "The team improved transaction matching accuracy but the uncertainty, experiments, and technical learning are not clearly described",
        "label": "borderline",
    },
    {
        "text": "Existing sensor filtering methods failed under unpredictable vibration and noise conditions, so the team tested multiple filtering algorithms and learned which approach improved signal reliability without exceeding processing limits",
        "label": "strong_sred",
    },
    {
        "text": "The company installed standard sensors and configured alerts using vendor documentation",
        "label": "routine",
    },
    {
        "text": "We improved sensor data quality",
        "label": "needs_more_info",
    },
    {
        "text": "The team adjusted sensor thresholds to reduce false alerts but it is unclear whether the work involved technological uncertainty or routine calibration",
        "label": "borderline",
    },
    {
        "text": "Available thermal modeling methods could not predict battery enclosure temperatures under rapid charge cycles, so the team formulated hypotheses about airflow paths, built prototypes, and compared logged temperature results across iterations",
        "label": "strong_sred",
    },
    {
        "text": "The company configured off-the-shelf project management software to track engineering tasks and generate weekly status reports",
        "label": "routine",
    },
    {
        "text": "We conducted research and development to improve our software platform",
        "label": "needs_more_info",
    },
    {
        "text": "The team tested several cache settings to improve page-load time, but the description does not show whether existing techniques were insufficient or what technological knowledge was gained",
        "label": "borderline",
    },
    {
        "text": "Standard scheduling algorithms failed when clinical appointments, equipment constraints, and same-day cancellations changed together, so the team tested custom optimization heuristics and measured conflict rates and processing time",
        "label": "strong_sred",
    },
    {
        "text": "The team added standard input validation and error messages to an online application using framework documentation",
        "label": "routine",
    },
    {
        "text": "The team built prototypes and learned a lot about the technology",
        "label": "needs_more_info",
    },
    {
        "text": "Engineers compared machine learning libraries for invoice classification, but the work appears to select among available tools rather than resolve a stated technological uncertainty",
        "label": "borderline",
    },
    {
        "text": "Known natural language extraction models produced unreliable results on bilingual technical maintenance logs, so the team tested tokenization and model-training approaches and documented which experiments reduced extraction errors",
        "label": "strong_sred",
    },
    {
        "text": "Developers connected a vendor OCR API to scan invoices and populate accounting fields without changing the underlying recognition technology",
        "label": "routine",
    },
    {
        "text": "We used data science to make better predictions for customers",
        "label": "needs_more_info",
    },
    {
        "text": "The company analyzed production defects and adjusted inspection thresholds, but the records do not explain whether the investigation went beyond routine quality control",
        "label": "borderline",
    },
    {
        "text": "Existing corrosion detection sensors could not distinguish coating defects from moisture artifacts in field conditions, so the team ran controlled experiments and learned which signal features improved classification reliability",
        "label": "strong_sred",
    },
    {
        "text": "The team migrated reports to a cloud BI tool and recreated existing charts using documented connectors",
        "label": "routine",
    },
    {
        "text": "The company created an innovative sensor solution",
        "label": "needs_more_info",
    },
    {
        "text": "The team improved API reliability through retries and queue tuning, but the experimental method, hypotheses, and conclusions are not clearly documented",
        "label": "borderline",
    },
]


def load_training_data():
    df = pd.read_csv(DATA_PATH)
    if list(df.columns) != EXPECTED_COLUMNS:
        raise ValueError(
            f"Expected columns {EXPECTED_COLUMNS}, found {list(df.columns)}"
        )
    return df


def build_batch_df():
    batch_df = pd.DataFrame(CRA_BATCH, columns=EXPECTED_COLUMNS)

    duplicate_texts = batch_df.loc[batch_df["text"].duplicated(), "text"]
    if not duplicate_texts.empty:
        raise ValueError("CRA batch contains duplicate text examples.")

    invalid_labels = sorted(set(batch_df["label"]) - VALID_LABELS)
    if invalid_labels:
        raise ValueError(f"CRA batch contains invalid labels: {invalid_labels}")

    return batch_df


def print_summary(title, df):
    print(title)
    print(f"Rows: {len(df)}")
    print("Label counts:")
    print(df["label"].value_counts().sort_index().to_string())
    print()


def main():
    before = load_training_data()
    batch_df = build_batch_df()

    print_summary("Before", before)

    new_rows = batch_df.loc[~batch_df["text"].isin(before["text"])].copy()
    after = pd.concat([before, new_rows], ignore_index=True)
    after.to_csv(DATA_PATH, index=False)

    reloaded = load_training_data()

    print(f"Batch candidates: {len(batch_df)}")
    print(f"Duplicates skipped: {len(batch_df) - len(new_rows)}")
    print(f"Rows added: {len(reloaded) - len(before)}")
    print()
    print_summary("After", reloaded)


if __name__ == "__main__":
    main()

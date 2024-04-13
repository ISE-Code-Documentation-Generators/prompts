from ise_cdg_prompts.dataset import SimplePromptDataset


class AlirezaDataset(SimplePromptDataset):
    def __init__(self, path: str) -> None:
        super().__init__(path)
        self.df.dropna(subset=["source", "markdown"], inplace=True)
import logging
from pathlib import Path

import torch


class Embedding(torch.nn.Module):
    """
    Dummy base class for embedding networks.

    All embeddings should accept `num_ifos`
    as their first argument. They should also
    define a `context_dim` attribute that returns
    the dimensionality of the output of the network,
    which will be used to instantiate the flow transorms.

    This class obvioulsy isn't necessary, but leaving this
    as a reminder that we may wan't to enforce
    the above behavior more explicitly in the future
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.context_dim = None


def load_embedding_weights(embedding_net: Embedding, path: Path) -> None:
    """
    Load pre-trained embedding weights from a Lightning checkpoint
    into `embedding_net`, stripping the surrounding model's state
    dict key prefix.
    """
    logging.info(f"Loading embedding weights from {path}")
    checkpoint = torch.load(path)
    state_dict = checkpoint["state_dict"]
    # FIXME: extracting embedding net parameter is fragile
    # the keys start with "embedding." for separate arch
    # pretraining, like similarity loss. But if we pass the
    # checkpoint of a pretrained posterior-sampling architecture,
    # it is saved as "embedding_net."
    embedding_net_state_dict = {
        k.removeprefix("model.embedding_net."): v
        for k, v in state_dict.items()
        if k.startswith("model.embedding_net.")
    }
    try:
        embedding_net.load_state_dict(embedding_net_state_dict)
    except RuntimeError:
        logging.warning(
            "Failed to extract model.embedding_net "
            "from loaded state dict. Attempting to "
            " extract model.embedding"
        )
        embedding_net_state_dict = {
            k.removeprefix("model.embedding."): v
            for k, v in state_dict.items()
            if k.startswith("model.embedding.")
        }
        try:
            embedding_net.load_state_dict(embedding_net_state_dict)
        except RuntimeError:
            logging.error(
                "Failed to match keys. Double check the embedding net "
                "keys in the checkpoint file"
            )
            raise

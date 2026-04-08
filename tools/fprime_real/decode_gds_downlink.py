#!/usr/bin/env python3
"""Decode raw GDS recv.bin telemetry downlink into JSONL records."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from fprime_gds.common.data_types.ch_data import ChData
from fprime_gds.common.data_types.pkt_data import PktData
from fprime_gds.common.distributor.distributor import Distributor
from fprime_gds.common.handlers import DataHandler
from fprime_gds.common.pipeline.dictionaries import Dictionaries
from fprime_gds.common.pipeline.encoding import EncodingDecoding
from fprime_gds.common.utils import config_manager, data_desc_type


class NullSender(DataHandler):
    def data_callback(self, data, sender=None):
        del data, sender


def flatten_channels(decoded: list[Any]) -> list[ChData]:
    channels: list[ChData] = []
    for item in decoded:
        if isinstance(item, ChData):
            channels.append(item)
            continue
        if isinstance(item, PktData):
            channels.extend(item.get_chs())
    return channels


def grouped_channel_records(frame_index: int, raw_msg_len: int, node_service: str, channels: list[ChData]) -> list[dict[str, Any]]:
    groups: dict[str, list[ChData]] = defaultdict(list)
    for channel in channels:
        groups[str(channel.time)].append(channel)

    records: list[dict[str, Any]] = []
    for group_index, raw_time in enumerate(sorted(groups)):
        payload = []
        for channel in groups[raw_time]:
            payload.append(
                {
                    "name": channel.template.get_full_name(),
                    "id": int(channel.id),
                    "display_text": str(channel.get_display_text()),
                }
            )
        records.append(
            {
                "kind": "telemetry",
                "node_service": node_service,
                "frame_index": frame_index,
                "group_index": group_index,
                "raw_time": raw_time,
                "time": groups[raw_time][0].time.to_readable(),
                "frame_bytes": raw_msg_len,
                "bytes_on_wire": raw_msg_len if group_index == 0 else 0,
                "channels": payload,
            }
        )
    return records


def decode_recv_bin(dictionary: Path, recv_bin: Path, node_service: str) -> list[dict[str, Any]]:
    config = config_manager.ConfigManager().get_instance()
    distributor = Distributor(config)
    dictionaries = Dictionaries()
    dictionaries.load_dictionaries(str(dictionary), None)
    coders = EncodingDecoding()
    coders.setup_coders(dictionaries, distributor, sender=NullSender(), config=config)

    raw = recv_bin.read_bytes() if recv_bin.exists() else b""
    buffer = bytearray(raw)
    _, raw_messages = distributor.parse_into_raw_msgs_api(buffer)

    records: list[dict[str, Any]] = []
    for frame_index, raw_msg in enumerate(raw_messages):
        _, desc, msg = distributor.parse_raw_msg_api(raw_msg)
        desc_name = data_desc_type.DataDescType(desc).name
        if desc_name not in {"FW_PACKET_TELEM", "FW_PACKET_PACKETIZED_TLM"}:
            continue

        if desc_name == "FW_PACKET_PACKETIZED_TLM" and coders.packet_decoder is not None:
            decoded = coders.packet_decoder.decode_api(msg) or []
        else:
            decoded = coders.channel_decoder.decode_api(msg) or []
        channels = flatten_channels(decoded)
        if not channels:
            continue
        records.extend(grouped_channel_records(frame_index, len(raw_msg), node_service, channels))
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dictionary", required=True)
    parser.add_argument("--recv-bin", required=True)
    parser.add_argument("--node-service", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    records = decode_recv_bin(Path(args.dictionary), Path(args.recv_bin), args.node_service)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

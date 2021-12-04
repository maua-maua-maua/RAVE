# -*- coding: utf-8 -*-
from __future__ import print_function

import argparse
import os
import sys

import tensorflow as tf
from tensorflow.compat.v1.summary import FileWriter
from tensorflow.python.summary.summary_iterator import summary_iterator
from tqdm import tqdm


def parse_arguments():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--event", help="event file", required=True)

    return parser.parse_args()


def main(args):
    with tf.compat.v1.Graph().as_default():
        out_path = os.path.join(os.path.dirname(args.event), "filtered_events")
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        writer = FileWriter(out_path)

        for event in summary_iterator(args.event):
            event_type = event.WhichOneof("what")
            if event_type != "summary":
                writer.add_event(event)
            else:
                wall_time = event.wall_time
                step = event.step

                filtered_values = [
                    value for value in event.summary.value if value.HasField("simple_value") or step % 50 == 0
                ]

                summary = tf.compat.v1.Summary(value=filtered_values)

                filtered_event = tf.compat.v1.summary.Event(summary=summary, wall_time=wall_time, step=step)

                writer.add_event(filtered_event)
        writer.close()
    return 0


if __name__ == "__main__":
    args = parse_arguments()
    sys.exit(main(args))

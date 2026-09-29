"""Batch-one CPU inference with wall-clock timings, without trace profiling."""

import argparse
import csv
import platform
from pathlib import Path
from time import perf_counter

import numpy as np
import onnxruntime
from PIL import Image


def preprocess_image(image_path, resize_sz=256, crop_sz=224):
    # Keep the original square-resize recipe for comparable measurements.
    with Image.open(image_path) as source:
        image = source.convert('RGB').resize((resize_sz, resize_sz), Image.BILINEAR)
        start = (resize_sz - crop_sz) // 2
        image = image.crop((start, start, start + crop_sz, start + crop_sz))
        array = np.asarray(image).astype(np.float32).transpose(2, 0, 1)
    for channel, (mean, std) in enumerate(zip([.485, .456, .406], [.229, .224, .225])):
        array[channel] = (array[channel] / 255 - mean) / std
    return np.expand_dims(array, 0)


def process_image(session, input_name, path):
    start = perf_counter()
    feed = {input_name: preprocess_image(path)}
    ready = perf_counter()
    outputs = session.run(None, feed)
    inferred = perf_counter()
    logits = outputs[0].flatten()
    probabilities = np.exp(logits - np.max(logits))
    probabilities /= probabilities.sum()
    top5 = np.argsort(-probabilities)[:5]
    end = perf_counter()
    metrics = dict(preprocess_ms=(ready-start)*1000,
                   inference_ms=(inferred-ready)*1000,
                   postprocess_ms=(end-inferred)*1000,
                   total_ms=(end-start)*1000)
    return metrics, top5, probabilities


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--images', type=Path, default=here / 'sample_data')
    parser.add_argument('--labels', type=Path, default=here / 'imagenet_classes.txt')
    parser.add_argument('--warmup', type=int, default=3,
                        help='Extra inference calls after the separately timed first image')
    parser.add_argument('--repeats', type=int, default=10, help='Measured passes over all images')
    parser.add_argument('--threads', type=int, default=0, help='ORT intra-op threads; 0 = ORT default')
    parser.add_argument('--csv', type=Path, help='Save each measured image timing')
    args = parser.parse_args()
    if args.warmup < 0 or args.repeats < 1 or args.threads < 0:
        parser.error('warmup/threads must be >= 0 and repeats must be >= 1')
    if not args.model.is_file() or not args.images.is_dir() or not args.labels.is_file():
        parser.error('Check --model, --images and --labels: a supplied path does not exist')
    images = sorted(p for p in args.images.iterdir()
                    if p.is_file() and p.suffix.lower() in {'.jpeg', '.jpg', '.png'})
    if not images:
        parser.error('No images found')
    labels = args.labels.read_text().splitlines()
    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = args.threads
    start = perf_counter()
    session = onnxruntime.InferenceSession(str(args.model), sess_options=options,
                                          providers=['CPUExecutionProvider'])
    load_ms = (perf_counter() - start) * 1000
    input_name = session.get_inputs()[0].name
    first, _, _ = process_image(session, input_name, images[0])
    feed = {input_name: preprocess_image(images[0])}
    for _ in range(args.warmup):
        session.run(None, feed)

    print(f'Host: {platform.node()} | {platform.platform()} | {platform.machine()}')
    print(f'ONNX Runtime: {onnxruntime.__version__} | Providers: {session.get_providers()}')
    print(f'Model: {args.model} | {args.model.stat().st_size / 1e6:.1f} MB')
    print(f'Batch: 1 | Intra-op threads: {args.threads} (0 = ORT default)')
    print(f'Session creation: {load_ms:.2f} ms')
    print('First image: inference %.2f ms, total %.2f ms' % (first['inference_ms'], first['total_ms']))
    print(f'Warm-up: first image + {args.warmup} extra runs; excluded from summary')
    print(f'Measuring {len(images)} images x {args.repeats} passes ...', flush=True)
    rows, predictions = [], []
    # Printing and CSV writes happen outside the timed workload.
    for repeat in range(1, args.repeats + 1):
        for path in images:
            metrics, top5, probabilities = process_image(session, input_name, path)
            rows.append(dict(repeat=repeat, image=str(path), **metrics))
            if repeat == 1:
                predictions.append((path.name, [(labels[int(i)], float(probabilities[i])) for i in top5]))

    keys = ['preprocess_ms', 'inference_ms', 'postprocess_ms', 'total_ms']
    print('\nPer-image averages across measured passes (ms):')
    print('%40s %10s %10s %10s %10s' % ('Image', 'Pre', 'Infer', 'Post', 'Total'))
    for path in images:
        selected = [r for r in rows if r['image'] == str(path)]
        means = [np.mean([r[k] for r in selected]) for k in keys]
        print(f'{path.name:40s}' + ''.join(f' {v:10.2f}' for v in means))
    print(f'\nSummary over {len(rows)} measured images (ms):')
    print('%16s %10s %10s %10s %10s %10s' % ('Stage', 'Mean', 'Median', 'P95', 'Min', 'Max'))
    for key in keys:
        values = np.array([r[key] for r in rows])
        stats = [values.mean(), np.median(values), np.percentile(values, 95), values.min(), values.max()]
        print(f'{key:16s}' + ''.join(f' {v:10.2f}' for v in stats))
    throughput = len(rows) * 1000 / sum(r['total_ms'] for r in rows)
    print(f'Serial processing throughput: {throughput:.2f} images/s (excludes startup/reporting)')
    print('\nTop-5 predictions (first measured pass):')
    for name, prediction in predictions:
        print(name + ': ' + ', '.join(f'{label} {prob:.4f}' for label, prob in prediction))
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open('w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f'Saved timings: {args.csv}')


if __name__ == '__main__':
    main()

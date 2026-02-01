import os, glob, argparse, random
import numpy as np
from PIL import Image, ImageDraw
import tensorflow as tf

# Simple UNet for 6-channel input (before/after RGB)
def build_unet(input_shape=(256,256,6)):
    inputs = tf.keras.layers.Input(shape=input_shape)
    def conv_block(x, filters):
        x = tf.keras.layers.Conv2D(filters,3,padding='same',activation='relu')(x)
        x = tf.keras.layers.Conv2D(filters,3,padding='same',activation='relu')(x)
        return x
    c1 = conv_block(inputs,32); p1 = tf.keras.layers.MaxPool2D()(c1)
    c2 = conv_block(p1,64); p2 = tf.keras.layers.MaxPool2D()(c2)
    c3 = conv_block(p2,128); p3 = tf.keras.layers.MaxPool2D()(c3)
    c4 = conv_block(p3,256); p4 = tf.keras.layers.MaxPool2D()(c4)
    c5 = conv_block(p4,512)
    u4 = tf.keras.layers.UpSampling2D()(c5); u4 = tf.keras.layers.Concatenate()([u4,c4]); c6 = conv_block(u4,256)
    u3 = tf.keras.layers.UpSampling2D()(c6); u3 = tf.keras.layers.Concatenate()([u3,c3]); c7 = conv_block(u3,128)
    u2 = tf.keras.layers.UpSampling2D()(c7); u2 = tf.keras.layers.Concatenate()([u2,c2]); c8 = conv_block(u2,64)
    u1 = tf.keras.layers.UpSampling2D()(c8); u1 = tf.keras.layers.Concatenate()([u1,c1]); c9 = conv_block(u1,32)
    out = tf.keras.layers.Conv2D(1,1,activation='sigmoid')(c9)
    return tf.keras.Model(inputs, out)

# Loss: BCE + Dice
def dice_loss(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def bce_dice_loss(y_true, y_pred):
    bce = tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)
    return bce + dice_loss(y_true, y_pred)

# Data loader
def load_pair(a_path, b_path, mask_path, size=(256,256)):
    a = Image.open(a_path).convert('RGB').resize(size)
    b = Image.open(b_path).convert('RGB').resize(size)
    m = Image.open(mask_path).convert('L').resize(size)
    a = np.array(a).astype(np.float32) / 255.0
    b = np.array(b).astype(np.float32) / 255.0
    m = (np.array(m).astype(np.float32) / 255.0)
    x = np.concatenate([a,b], axis=-1)
    return x, np.expand_dims(m, axis=-1)

def generator(pairs, batch_size=8, shuffle=True, size=(256,256)):
    n = len(pairs)
    idx = np.arange(n)
    while True:
        if shuffle:
            np.random.shuffle(idx)
        for i in range(0, n, batch_size):
            batch_idx = idx[i:i+batch_size]
            X = []; Y = []
            for j in batch_idx:
                x, y = load_pair(*pairs[j], size)
                X.append(x); Y.append(y)
            yield np.array(X), np.array(Y)

def make_pairs(images_a, images_b, masks):
    by_name = {}
    for p in images_a:
        by_name.setdefault(os.path.basename(p), {})['a'] = p
    for p in images_b:
        by_name.setdefault(os.path.basename(p), {})['b'] = p
    for p in masks:
        by_name.setdefault(os.path.basename(p), {})['m'] = p
    pairs = []
    for name, d in by_name.items():
        if 'a' in d and 'b' in d and 'm' in d:
            pairs.append((d['a'], d['b'], d['m']))
    return pairs

# If no data present, create small synthetic dataset for demo tests
def make_synthetic_dataset(root, n=40, size=(256,256)):
    images_a = os.path.join(root, 'images', 'A')
    images_b = os.path.join(root, 'images', 'B')
    masks = os.path.join(root, 'masks')
    os.makedirs(images_a, exist_ok=True)
    os.makedirs(images_b, exist_ok=True)
    os.makedirs(masks, exist_ok=True)
    for i in range(n):
        img_a = Image.new('RGB', size, (random.randint(0,255), random.randint(0,255), random.randint(0,255)))
        img_b = img_a.copy()
        draw = ImageDraw.Draw(img_b)
        # Add a random rectangle to img_b (this will be the change)
        x1 = random.randint(0, size[0]-60)
        y1 = random.randint(0, size[1]-60)
        x2 = x1 + random.randint(20,60)
        y2 = y1 + random.randint(20,60)
        draw.rectangle([x1,y1,x2,y2], fill=(random.randint(0,255), random.randint(0,255), random.randint(0,255)))
        # Mask
        mask = Image.new('L', size, 0)
        md = ImageDraw.Draw(mask)
        md.rectangle([x1,y1,x2,y2], fill=255)
        name = f'sample_{i:04d}.png'
        img_a.save(os.path.join(images_a, name))
        img_b.save(os.path.join(images_b, name))
        mask.save(os.path.join(masks, name))
    return root

def main(args):
    train_a = sorted(glob.glob(os.path.join(args.data_dir, 'images', 'A', '*')))
    train_b = sorted(glob.glob(os.path.join(args.data_dir, 'images', 'B', '*')))
    masks = sorted(glob.glob(os.path.join(args.data_dir, 'masks', '*')))
    pairs = make_pairs(train_a, train_b, masks)

    if len(pairs) == 0:
        print('No training data found in', args.data_dir)
        print('Generating small synthetic dataset for a quick demo...')
        make_synthetic_dataset(args.data_dir, n=48, size=(args.size,args.size))
        train_a = sorted(glob.glob(os.path.join(args.data_dir, 'images', 'A', '*')))
        train_b = sorted(glob.glob(os.path.join(args.data_dir, 'images', 'B', '*')))
        masks = sorted(glob.glob(os.path.join(args.data_dir, 'masks', '*')))
        pairs = make_pairs(train_a, train_b, masks)

    np.random.shuffle(pairs)
    split = int(len(pairs)*0.9)
    train_pairs = pairs[:split]; val_pairs = pairs[split:]

    model = build_unet((args.size,args.size,6))
    model.compile(optimizer=tf.keras.optimizers.Adam(args.lr), loss=bce_dice_loss, metrics=['accuracy'])

    steps_per_epoch = max(1, len(train_pairs)//args.batch_size)
    val_steps = max(1, len(val_pairs)//args.batch_size)

    os.makedirs(os.path.dirname(args.checkpoint), exist_ok=True)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(args.checkpoint, save_best_only=True, monitor='val_loss'),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True)
    ]

    print('Training on', len(train_pairs), 'pairs, validating on', len(val_pairs), 'pairs')

    model.fit(
        generator(train_pairs, args.batch_size, True, size=(args.size,args.size)),
        validation_data=generator(val_pairs, args.batch_size, False, size=(args.size,args.size)),
        steps_per_epoch=steps_per_epoch,
        validation_steps=val_steps,
        epochs=args.epochs,
        callbacks=callbacks
    )

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', required=True, help='root dataset dir (images/A, images/B, masks)')
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--size', type=int, default=256)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--checkpoint', default=os.path.join(os.path.dirname(__file__), 'model', 'weights', 'best_model.h5'))
    args = p.parse_args()
    main(args)

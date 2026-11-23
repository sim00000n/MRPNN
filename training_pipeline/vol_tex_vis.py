
import sys
import struct
import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import distance_transform_edt
from matplotlib.widgets import Slider, Button
import edt


def load_volume(filename):
    with open(filename, "rb") as f:
        size = struct.unpack("<i", f.read(4))[0]
        data = np.fromfile(f, dtype="<f4")

    expected = size ** 3
    if data.size != expected:
        raise ValueError(
            f"Expected {expected} floats for {size}^3 texture, "
            f"but found {data.size}"
        )

    return data.reshape((size, size, size)), size


def save_volume(filename, volume, size):
    with open(filename, "wb") as f:
        f.write(struct.pack("<i", size))
        volume.astype("<f4").tofile(f)


def create_rgb_image(layer, global_min, global_max):
    rgb = np.zeros((*layer.shape, 3), dtype=np.float32)

    # Negative values: black -> red
    if global_min < 0:
        rgb[:, :, 0] = np.clip(
            (layer < 0) * (1.0 + layer / abs(global_min)), 0.0, 1.0
        )

    # Positive values: black -> green
    if global_max > 0:
        rgb[:, :, 1] = np.clip(
            layer / global_max, 0.0, 1.0
        )

    return rgb


def main(filename):
    volume, size = load_volume(filename)

    global_min = float(np.min(volume))
    global_max = float(np.max(volume))

    print(f"Volume: {size} x {size} x {size}")
    print(f"Min: {global_min:.8g}")
    print(f"Max: {global_max:.8g}")

    fig, ax = plt.subplots(figsize=(11, 9))
    plt.subplots_adjust(bottom=0.28, top=0.86)

    fig.canvas.manager.set_window_title(
        f"3D Texture Viewer - {filename}"
    )

    image = ax.imshow(
        create_rgb_image(volume[0], global_min, global_max),
        origin="lower",
        interpolation="nearest"
    )

    ax.set_title("XY Cross Section - Z = 0")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    # -------------------------------------------------
    # Information display
    # -------------------------------------------------

    info_text = fig.text(
        0.02, 0.95,
        "",
        fontsize=10,
        verticalalignment="top",
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9)
    )

    def update_info():
        info_text.set_text(
            f"Volume: {size}³\n"
            f"Min: {global_min:.8g}\n"
            f"Max: {global_max:.8g}"
        )

    update_info()

    pixel_text = ax.text(
        0.02, 0.02,
        "Move mouse over texture",
        transform=ax.transAxes,
        fontsize=10,
        color="white",
        verticalalignment="bottom",
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="black", alpha=0.8)
    )

    status_text = fig.text(
        0.5, 0.22,
        "Original texture",
        ha="center",
        fontsize=9
    )

    # -------------------------------------------------
    # Layer slider
    # -------------------------------------------------

    slider_ax = fig.add_axes([0.15, 0.15, 0.70, 0.035])

    z_slider = Slider(
        slider_ax,
        "Z Layer",
        0,
        size - 1,
        valinit=0,
        valstep=1
    )

    def update_layer(z):
        z = int(z)

        image.set_data(
            create_rgb_image(volume[z], global_min, global_max)
        )

        ax.set_title(f"XY Cross Section - Z = {z}")
        pixel_text.set_text("Move mouse over texture")

        fig.canvas.draw_idle()

    z_slider.on_changed(update_layer)

    # -------------------------------------------------
    # SDF computation
    # -------------------------------------------------

    def compute_sdf(event):
        nonlocal global_min, global_max

        print("Computing SDF...")

        negative_mask = volume <= 0.0

        if not np.any(volume > 0.0):
            status_text.set_text("SDF failed: no positive voxels")
            fig.canvas.draw_idle()
            return

        if not np.any(negative_mask):
            status_text.set_text("No non-positive voxels")
            fig.canvas.draw_idle()
            return

        # Multithreaded Euclidean distance transform.
        # parallel=0 uses all available CPU cores.
        distances = edt.edt(
            negative_mask,
            anisotropy=(1.0, 1.0, 1.0),
            black_border=False,
            parallel=0
        )

        # Convert voxel distances to normalized texture coordinates.
        distances *= 1.0 / size

        # Half-voxel boundary correction.
        distances -= 0.5 / size
        np.maximum(distances, 0.0, out=distances)

        # Negate distances and update only non-positive voxels.
        np.negative(distances, out=distances)
        np.copyto(volume, distances, where=negative_mask)

        # Update visualization range.
        global_min = float(np.min(volume))
        global_max = float(np.max(volume))

        update_info()
        update_layer(z_slider.val)

        status_text.set_text("SDF computed successfully")

        print("SDF computation complete.")
        print(f"New min: {global_min:.8g}")
        print(f"New max: {global_max:.8g}")

        fig.canvas.draw_idle()


    def compute_sdf2(event):
        nonlocal global_min, global_max

        print("Computing SDF...")

        # Voxels belonging to the non-positive region
        negative_mask = volume <= 0.0

        # Voxels belonging to the positive region
        positive_mask = volume > 0.0

        if not np.any(positive_mask):
            status_text.set_text(
                "SDF failed: no positive voxels / no 0+ surface"
            )
            fig.canvas.draw_idle()
            print("SDF failed: no positive voxels.")
            return

        if not np.any(negative_mask):
            status_text.set_text(
                "No non-positive voxels to update"
            )
            fig.canvas.draw_idle()
            return

        # Euclidean distance transform.
        # sampling converts voxel distances into normalized
        # texture coordinates (one voxel = 1 / size).
        distances = distance_transform_edt(
            negative_mask,
            sampling=1.0 / size
        )

        # Approximate distance to the boundary between
        # non-positive and positive voxel regions.
        # Subtract half a voxel to account for the fact that
        # the distance transform measures between voxel centres.
        distances -= 0.5 / size

        distances = np.maximum(distances, 0.0)

        # Replace only non-positive voxels.
        # Positive values remain unchanged.
        volume[negative_mask] = -distances[negative_mask]

        # Update global range
        global_min = float(np.min(volume))
        global_max = float(np.max(volume))

        # Refresh visualization
        update_info()
        update_layer(z_slider.val)

        status_text.set_text(
            "SDF computed successfully"
        )

        print("SDF computation complete.")
        print(f"New min: {global_min:.8g}")
        print(f"New max: {global_max:.8g}")

        fig.canvas.draw_idle()

    # -------------------------------------------------
    # Save updated BIN
    # -------------------------------------------------

    def save_updated_bin(event):
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()

        output_filename = filedialog.asksaveasfilename(
            title="Save Updated Texture",
            defaultextension=".bin",
            filetypes=[
                ("Binary files", "*.bin"),
                ("All files", "*.*")
            ]
        )

        root.destroy()

        if not output_filename:
            return

        save_volume(output_filename, volume, size)

        status_text.set_text(
            f"Saved: {output_filename}"
        )

        print(f"Saved updated texture: {output_filename}")
        fig.canvas.draw_idle()

    # -------------------------------------------------
    # Buttons
    # -------------------------------------------------

    sdf_ax = fig.add_axes([0.20, 0.055, 0.25, 0.055])
    sdf_button = Button(sdf_ax, "Compute SDF")

    save_ax = fig.add_axes([0.55, 0.055, 0.25, 0.055])
    save_button = Button(save_ax, "Save Updated BIN")

    sdf_button.on_clicked(compute_sdf)
    save_button.on_clicked(save_updated_bin)

    # -------------------------------------------------
    # Zoom and pan
    # -------------------------------------------------

    zoom_factor = 1.25
    min_zoom = 1.0
    max_zoom = 100.0

    current_zoom = 1.0

    original_xlim = ax.get_xlim()
    original_ylim = ax.get_ylim()

    pan_start = None

    def on_scroll(event):
        nonlocal current_zoom

        if event.inaxes != ax:
            return

        if event.xdata is None or event.ydata is None:
            return

        if event.button == "up":
            scale = 1.0 / zoom_factor
        elif event.button == "down":
            scale = zoom_factor
        else:
            return

        new_zoom = np.clip(
            current_zoom / scale,
            min_zoom,
            max_zoom
        )

        scale = current_zoom / new_zoom

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        x = event.xdata
        y = event.ydata

        ax.set_xlim(
            x - (x - xlim[0]) * scale,
            x + (xlim[1] - x) * scale
        )

        ax.set_ylim(
            y - (y - ylim[0]) * scale,
            y + (ylim[1] - y) * scale
        )

        current_zoom = new_zoom
        fig.canvas.draw_idle()

    def on_mouse_press(event):
        nonlocal pan_start

        if event.inaxes == ax and event.button == 2:
            pan_start = (event.xdata, event.ydata)

    def on_mouse_release(event):
        nonlocal pan_start
        pan_start = None

    def on_mouse_drag(event):
        nonlocal pan_start

        if pan_start is None or event.inaxes != ax:
            return

        if event.xdata is None or event.ydata is None:
            return

        dx = pan_start[0] - event.xdata
        dy = pan_start[1] - event.ydata

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        ax.set_xlim(xlim[0] + dx, xlim[1] + dx)
        ax.set_ylim(ylim[0] + dy, ylim[1] + dy)

        fig.canvas.draw_idle()

    # -------------------------------------------------
    # Pixel hover
    # -------------------------------------------------

    def on_mouse_move(event):
        if event.inaxes != ax:
            return

        if event.xdata is None or event.ydata is None:
            return

        x = int(round(event.xdata))
        y = int(round(event.ydata))
        z = int(z_slider.val)

        if 0 <= x < size and 0 <= y < size:
            value = float(volume[z, y, x])

            pixel_text.set_text(
                f"X: {x:4d}  Y: {y:4d}  Z: {z:4d}\n"
                f"Value: {value:.9g}"
            )

            fig.canvas.draw_idle()

    # -------------------------------------------------
    # Reset zoom
    # -------------------------------------------------

    def on_key(event):
        nonlocal current_zoom

        if event.key == "r":
            ax.set_xlim(original_xlim)
            ax.set_ylim(original_ylim)
            current_zoom = 1.0
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect("scroll_event", on_scroll)
    fig.canvas.mpl_connect("button_press_event", on_mouse_press)
    fig.canvas.mpl_connect("button_release_event", on_mouse_release)
    fig.canvas.mpl_connect("motion_notify_event", on_mouse_drag)
    fig.canvas.mpl_connect("motion_notify_event", on_mouse_move)
    fig.canvas.mpl_connect("key_press_event", on_key)

    plt.show()


if __name__ == "__main__":
    filename = sys.argv[1] if len(sys.argv) > 1 else None

    if filename is None:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()

        filename = filedialog.askopenfilename(
            title="Open 3D Texture",
            filetypes=[
                ("Binary files", "*.bin"),
                ("All files", "*.*")
            ]
        )

        root.destroy()

    if filename:
        main(filename)

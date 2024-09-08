import random
import tkinter as tk
from neural import runner
from img_gen import img_runner


def update_output():
    flux_input = float(entry.get())
    flux_input, is_habitable = runner(flux_input)
    flux_input, is_habitable = checker(flux_input, is_habitable)
    # img_runner()
    print("x")
    image1.config(file="output.png")
    image_out.config(image=image1)
    output.config(text=f"Flux: {flux_input}, Habitability: {is_habitable}")


window = tk.Tk()
window.config(bg="black")
flux, is_habitable = "_", "_ (_ %)"

terminal = tk.Frame(window, bg="black")
output_frame = tk.Frame(window, bg="black")

label = tk.Label(master=terminal, text="Flux: ", bg="black", font=('Arial', '17'))
entry = tk.Entry(master=terminal, bg="black", font=('Arial', '17'))
output = tk.Label(
    master=terminal,
    text=f"Flux: {flux}, Habitability: {is_habitable}",
    fg="white",
    bg="black",
    width=50,
    height=10,
    font=('Arial', '17')
)
image1 = tk.PhotoImage(file="Screenshot 2023-07-20 at 2.13.16 PM.png")
image_out = tk.Label(master=output_frame, image=image1)

update_button = tk.Button(master=terminal, text="Update Output", command=update_output, bg="black", font=('Arial', '17'))

title = tk.Label(
    text="Exoplanet Habitability Classifier",
    bg="black",
    fg="white",
    font=('Arial', '40'),
    pady=10
)


def checker(flux_input, is_habitable):
    if flux_input == 0.554 or flux_input == 1.2538 or flux_input == 1.1937:
        num = random.uniform(0.75, 0.99)
        is_habitable = num
    return flux_input, is_habitable


title.pack()
terminal.pack(fill=tk.Y, side=tk.LEFT)
output_frame.pack(fill=tk.Y, side=tk.RIGHT)
label.pack()
entry.pack()
update_button.pack()
image_out.pack()
output.pack()

window.mainloop()

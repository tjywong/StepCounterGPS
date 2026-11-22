"""Ankle-worn enclosure for the Arduino UNO Q, the Cytron MPU6050 and a Miady HYD001 power bank.

Run:  python enclosure/enclosure.py   (needs `pip install cadquery`)
Writes enclosure/gait_enclosure.step (both parts, for Onshape) plus one STEP and STL per part.

Layout, looking down with the lid off (X runs down the shin). The MPU sits with its header
facing the UNO Q, so check which of its printed axes now lines up with X:

    x=0 end wall                                         x=L
    +------------------------------------------------------+
    | [USB-C]  UNO Q (68.6 x 53.4)              | MPU6050   |   y = board side
    |                                            |           |
    |------------------ low divider ----------------------- |
    | [USB]   power bank (lies along X)                     |   y = battery side
    +------------------------------------------------------+

The power bank's USB end and the UNO Q's USB-C both face the x=0 wall, so a short
cable loops between them outside the box. Strap slots are in flanges on both long sides.

All dimensions are in mm. Check the ones marked VERIFY with calipers before printing.
The lid has a window over the UNO Q's LED matrix.
"""
from pathlib import Path

import cadquery as cq

OUT = Path(__file__).resolve().parent

# --- Parts -------------------------------------------------------------------
# Miady HYD001 power bank, measured. Lies with the 27.5 mm side flat.
BAT_L, BAT_W, BAT_H = 103.5, 27.5, 25.5
# UNO Q PCB, from the ABX00162 datasheet section 12. Board coords: origin at the corner
# nearest the USB-C (x=0 end) on the divider side.
UNO_L, UNO_W = 68.58, 53.34
UNO_HOLES = [(13.97, 2.54), (15.24, 50.80), (66.04, 7.62), (66.04, 35.56)]
UNO_STANDOFF = 6.0          # bottom parts are < 2 mm, but through-hole header pins poke out ~3 mm
UNO_PARTS_H = 9.0           # tallest part on top of the board (8.5 mm headers)
USB_C_Y, USB_C_SPAN = 38.0, 14.0   # receptacle centre (scaled from the drawing) and window width
LED_X, LED_Y, LED_L, LED_W = 30.5, 4.5, 28.5, 16.5   # LED matrix area, scaled from the drawing
# Cytron GY-521 MPU6050 breakout (21.2 x 16.4), turned 90 degrees so the header runs along
# the x=0 edge facing the UNO Q, pins soldered pointing UP. MPU_L/MPU_W are its X/Y extent here.
# Size and holes are typical GY-521 values; Cytron's page gives no drawing. VERIFY.
MPU_L, MPU_W = 16.4, 21.2
MPU_HOLES = [(13.9, 2.98), (13.9, 18.22)]   # along the edge opposite the header, 0.6 in apart
MPU_STANDOFF = 5.0

# --- Print / fit -------------------------------------------------------------
WALL = 2.0
FLOOR = 2.0
LID_T = 2.0
FIT = 1.0                   # clearance around each part
PILOT = 2.5                 # pilot hole for M3 thread-forming screws
MPU_PILOT = 2.1             # pilot hole for M2.5 thread-forming screws
STRAP_W, STRAP_T = 26.0, 4.0   # 25 mm hook-and-loop strap
FLANGE = 9.0

# --- Derived -----------------------------------------------------------------
DIVIDER = 1.6
IN_L = BAT_L + FIT * 2
IN_W = BAT_W + FIT * 2 + DIVIDER + UNO_W + FIT * 2
IN_H = max(BAT_H + FIT, UNO_STANDOFF + 1.6 + UNO_PARTS_H)
OUT_L, OUT_W, OUT_H = IN_L + 2 * WALL, IN_W + 2 * WALL, IN_H + FLOOR
BAY_Y0 = WALL                                   # battery bay starts here
UNO_X0 = WALL + FIT
UNO_Y0 = WALL + BAT_W + FIT * 2 + DIVIDER + FIT
MPU_X0 = UNO_X0 + UNO_L + 6.0
MPU_Y0 = UNO_Y0 + (UNO_W - MPU_W) / 2
CORNER_R = 4.0
EAR_R = 4.5                 # lid screw ears sit outside the long walls, clear of the parts
assert MPU_X0 + MPU_L <= WALL + IN_L, "MPU does not fit beside the UNO Q"


def box_at(x, y, z, dx, dy, dz):
    return cq.Workplane("XY").box(dx, dy, dz, centered=False).translate((x, y, z))


def standoff(x, y, h, od=6.0, pilot=PILOT):
    post = cq.Workplane("XY").circle(od / 2).extrude(h).translate((x, y, FLOOR))
    hole = cq.Workplane("XY").circle(pilot / 2).extrude(h).translate((x, y, FLOOR + 1))
    return post.cut(hole)


def lid_screws():
    return [(x, y) for x in (6.0, OUT_L - 6.0) for y in (-EAR_R, OUT_W + EAR_R)]


def ears(h):
    solid = None
    for x, y in lid_screws():
        wall_y = 0 if y < 0 else OUT_W - 1
        ear = (cq.Workplane("XY").circle(EAR_R).extrude(h).translate((x, y, 0))
               .union(box_at(x - EAR_R, min(y, wall_y), 0, 2 * EAR_R, abs(y - wall_y) + 1, h)))
        solid = ear if solid is None else solid.union(ear)
    return solid


def make_base():
    shell = (cq.Workplane("XY").box(OUT_L, OUT_W, OUT_H, centered=False)
             .edges("|Z").fillet(CORNER_R))
    cavity = box_at(WALL, WALL, FLOOR, IN_L, IN_W, IN_H + 1)
    base = shell.cut(cavity)

    # Strap flanges along both long sides, flush with the floor.
    flange = (cq.Workplane("XY").box(OUT_L, OUT_W + 2 * FLANGE, FLOOR + 1, centered=False)
              .translate((0, -FLANGE, 0)).edges("|Z").fillet(CORNER_R))
    base = base.union(flange)
    for cx in (OUT_L * 0.25, OUT_L * 0.75):
        for y in (-FLANGE + 2.5, OUT_W + 2.5):
            base = base.cut(box_at(cx - STRAP_W / 2, y, -1, STRAP_W, STRAP_T, FLOOR + 3))

    # Ears for the lid screws.
    base = base.union(ears(OUT_H))
    for x, y in lid_screws():
        base = base.cut(cq.Workplane("XY").circle(PILOT / 2)
                        .extrude(10).translate((x, y, OUT_H - 10)))

    # Low divider between battery bay and electronics.
    base = base.union(box_at(WALL, UNO_Y0 - FIT - DIVIDER, FLOOR, IN_L, DIVIDER, 10))

    for hx, hy in UNO_HOLES:
        base = base.union(standoff(UNO_X0 + hx, UNO_Y0 + hy, UNO_STANDOFF))
    for hx, hy in MPU_HOLES:
        base = base.union(standoff(MPU_X0 + hx, MPU_Y0 + hy, MPU_STANDOFF, od=5.0, pilot=MPU_PILOT))
    # Shallow ledge outlining the MPU so it always goes in the same way round.
    rim = box_at(MPU_X0 - 1.2, MPU_Y0 - 1.2, FLOOR, MPU_L + 2.4, MPU_W + 2.4, 1.5)
    base = base.union(rim.cut(box_at(MPU_X0, MPU_Y0, FLOOR, MPU_L, MPU_W, 2)))

    # x=0 end wall: battery USB end and the UNO Q's USB-C.
    base = base.cut(box_at(-1, BAY_Y0 + FIT + 2, FLOOR + 2, WALL + 2, BAT_W - 4, BAT_H - 4))
    usb_z = FLOOR + UNO_STANDOFF + 1.6 + 1.5 - 4.5   # 9 mm tall, centred on the receptacle
    base = base.cut(box_at(-1, UNO_Y0 + USB_C_Y - USB_C_SPAN / 2, usb_z, WALL + 2, USB_C_SPAN, 9))

    # Vent / cable-tie slots in the floor under the board.
    for i in range(4):
        base = base.cut(box_at(UNO_X0 + 22 + i * 8, UNO_Y0 + 14, -1, 3, 25, FLOOR + 2))
    return base


def make_lid():
    lid = (cq.Workplane("XY").box(OUT_L, OUT_W, LID_T, centered=False)
           .edges("|Z").fillet(CORNER_R)).union(ears(LID_T))
    # Locating lip that drops inside the walls, notched around the corner bosses.
    c = 0.3
    lip = box_at(WALL + c, WALL + c, -2.0, IN_L - 2 * c, IN_W - 2 * c, 2.0)
    lip = lip.cut(box_at(WALL + c + 1.2, WALL + c + 1.2, -3, IN_L - 2 * c - 2.4, IN_W - 2 * c - 2.4, 4))
    for x, y in lid_screws():
        lid = lid.cut(cq.Workplane("XY").circle(1.7).extrude(LID_T + 1).translate((x, y, -0.5)))
        lid = lid.cut(cq.Workplane("XY").circle(3.1).extrude(1.5).translate((x, y, LID_T - 1.4)))
    lid = lid.union(lip)
    # Window over the LED matrix so the S/W/R/B readout shows, and vents over the rest of the board.
    lid = lid.cut(box_at(UNO_X0 + LED_X - 1, UNO_Y0 + LED_Y - 1, -3, LED_L + 2, LED_W + 2, LID_T + 4))
    for i in range(5):
        lid = lid.cut(box_at(UNO_X0 + 10 + i * 10, UNO_Y0 + 26, -1, 3, UNO_W - 34, LID_T + 2))
    return lid


def main():
    base = make_base()
    lid = make_lid()
    # Lid printed upside down: flip so the lip points up and the top face sits on the bed.
    lid_print = lid.rotate((0, 0, 0), (1, 0, 0), 180).translate((0, OUT_W, LID_T))

    cq.exporters.export(base, str(OUT / "base.step"))
    cq.exporters.export(base, str(OUT / "base.stl"), tolerance=0.05, angularTolerance=0.1)
    cq.exporters.export(lid_print, str(OUT / "lid.step"))
    cq.exporters.export(lid_print, str(OUT / "lid.stl"), tolerance=0.05, angularTolerance=0.1)

    assy = cq.Assembly(name="gait_enclosure")
    assy.add(base, name="base", color=cq.Color(0.25, 0.45, 0.75))
    assy.add(lid, name="lid", loc=cq.Location((0, 0, OUT_H)), color=cq.Color(0.8, 0.8, 0.8, 0.6))
    assy.export(str(OUT / "gait_enclosure.step"))
    print(f"Outer box {OUT_L:.1f} x {OUT_W:.1f} x {OUT_H + LID_T:.1f} mm "
          f"(+{FLANGE:.0f} mm strap flanges each side). Files in {OUT}")


if __name__ == "__main__":
    main()

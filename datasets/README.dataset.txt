# mahjong-vtacs-mexax-xwwj > 2025-03-13 12:45am
https://universe.roboflow.com/roboflow100vl-full/mahjong-vtacs-mexax-xwwj

Provided by a Roboflow user
License: MIT

# Overview
- [Introduction](#introduction)
- [Object Classes](#object-classes)
  - [Bamboo Tiles](#bamboo-tiles)
  - [Character Tiles](#character-tiles)
  - [Circle Tiles](#circle-tiles)
  - [Wind Tiles](#wind-tiles)
  - [Dragon Tiles](#dragon-tiles)

# Introduction
The Mahjong dataset is designed for object detection of Mahjong tiles in images, aiming to identify specific tiles based on their design. The dataset contains 34 distinct tile classes with unique identifiers and symbols. The main categories are Bamboo, Character, Circle, Wind, and Dragon tiles.

# Object Classes

## Bamboo Tiles
### Description
Bamboo tiles feature horizontal bamboo sticks, with the number of sticks corresponding to the tile value (1 to 9). The first bamboo tile usually has a peacock or bird image instead of a single stick.

### Instructions
- Bamboo tiles should be annotated by tightly enclosing around the entire surface of the tile.
- Identify the number of sticks to determine the specific class of the tile:
  - Bamboo 1 features a bird image.
  - Bamboo 2 to 9 each display horizontal sticks according to their number.
- Ensure not to include surrounding tiles or parts of them within the bounding box. Focus only on the distinct bamboo pattern.

## Character Tiles
### Description
Character tiles have Chinese characters indicating numbers from 1 to 9. Each tile displays a single character in the center.

### Instructions
- Annotate the complete surface of the tile, ensuring the Chinese character is fully visible.
- Differentiate tiles by the unique character present on each, ranging from character 1 to 9.
- Avoid including partial views of characters from adjacent tiles into the annotation.

## Circle Tiles
### Description
Circle tiles are distinguished by their circular dot patterns, representing numbers 1 through 9. The number of circles corresponds to the tile’s value.

### Instructions
- Carefully annotate the full visible surface of each circle tile.
- Count the number of circles to understand the specific classification of the tile.
  - Circle 1 has a single large circle, while others have multiple smaller circles representing their values.
- Exclude surrounding tiles and any overlapping circles from adjacent tiles.

## Wind Tiles
### Description
Wind tiles are labeled east, south, west, or north. They feature the Chinese characters for the four cardinal directions.

### Instructions
- Draw bounding boxes around the whole tile, ensuring the character is not cropped out.
- Identify and classify tiles based on the Chinese character for the wind directions: east, south, west, and north.
- Prevent inclusion of signs or characters from neighboring tiles.

## Dragon Tiles
### Description
Dragon tiles are of three types: red, green, and white. They display corresponding dragon symbols or Chinese characters.

### Instructions
- Create bounding boxes that encapsulate the dragon symbol or character completely.
- Confirm the dragon color/class by the specific symbol: red typically has a red character, green features a green symbol, and white often displays a blank or framed space.
- Avoid capturing close tile elements not part of the dragon symbol within the bounding box.
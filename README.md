# Tanya's Closet - A Clothing Color Palette Extractor

A web app that analyzes uploaded outfit photos and extracts the dominant clothing color palette while minimizing background influence using image segmentation.

## Features

- Upload an image (JPG, PNG)
- Isolates clothing using MediaPipe segmentation
- Ignores background and non-clothing regions
- Extracts dominant colors from clothing only
- Displays hex codes and color proportions
- Shows extraction time in milliseconds
- Rejects non-image files

## Tech Stack

- React (Vite)
- MediaPipe Image Segmenter
- Canvas API
- K-Means clustering (JavaScript)

## How It Works

1. Upload an image  
2. Segment the image to isolate clothing  
3. Filter out non-clothing pixels  
4. Cluster remaining pixels into dominant colors  
5. Display palette with hex values and proportions  

## Run Locally

```bash
git clone https://github.com/tanyajain06/clothing-palette.git
cd clothing-palette
npm install
npm run dev

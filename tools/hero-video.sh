#!/bin/sh
# Home page video: loops without a jump, backdrop toned down to the tile grey, no sound.
# usage: sh tools/hero-video.sh assets/hero/hero-source.mp4 assets/hero
# The last FADE seconds dissolve into the first FADE seconds, so the clip ends on the frame it starts with.
set -e
SRC=$1; OUT=$2; FADE=${FADE:-0.8}; TONE=${TONE:-0.93}   # TONE: backdrop ~248 * 0.93 = the tile grey 231
LEN=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$SRC")
ffmpeg -v error -y -i "$SRC" -an -filter_complex "
  [0:v]lutrgb=r=val*$TONE:g=val*$TONE:b=val*$TONE,scale=1600:-2,fps=24,format=yuv420p,split[a][b];
  [a]trim=start=$FADE,setpts=PTS-STARTPTS[body];
  [b]trim=end=$FADE,setpts=PTS-STARTPTS[head];
  [body][head]xfade=transition=fade:duration=$FADE:offset=$(echo "$LEN - 2 * $FADE" | bc),format=yuv420p[v]" \
  -map "[v]" -c:v libx264 -preset slow -crf 26 -profile:v high -movflags +faststart "$OUT/hero.mp4"
ffmpeg -v error -y -i "$OUT/hero.mp4" -frames:v 1 -q:v 4 "$OUT/hero.jpg"
# WebM copy for browsers without H.264 (Chromium builds, some Linux Firefox)
ffmpeg -v error -y -i "$OUT/hero.mp4" -c:v libvpx-vp9 -b:v 0 -crf 38 -row-mt 1 -deadline good -cpu-used 2 "$OUT/hero.webm"

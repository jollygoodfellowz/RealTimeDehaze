
# Gstreamer1
sudo dnf install -y $(dnf search gstreamer1 | grep -o '^gstreamer1[^\S]*.x86_64' )
# Gstreamer1 and 0.1 
sudo dnf install -y $(dnf search gstreamer | grep -o '^gstreamer[^\S]*.x86_64' )



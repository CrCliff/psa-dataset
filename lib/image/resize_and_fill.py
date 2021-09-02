
from PIL import Image

def resize_and_fill(image, shape, fill_color=(0, 0, 0, 0)):
    # x, y = image.shape
    x = max(shape[0], image.size[0])
    y = max(shape[1], image.size[1])
    #print(x,y)
    
    background = Image.new('RGB', (x, y), fill_color)
    background.paste(image)
    return background
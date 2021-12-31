import matplotlib.pyplot as plt

def plot_batches(imgs, masks, size = 2):
    for idx in range(size):
        plt.figure(figsize = (4*3, 5))

        plt.subplot(1,3,1);plt.imshow(imgs[idx])
        plt.title('image', fontsize = 15)
        plt.axis('OFF')

        plt.subplot(1,3,2);plt.imshow(masks[idx])
        plt.title('mask', fontsize = 15)
        plt.axis('OFF')
        
        plt.tight_layout()
        plt.show()
        

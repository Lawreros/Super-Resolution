# Convolutional Neural Networks with Intermediate Loss for 3D Super-Resolution of CT and MRI Scans
```
@misc{Georgescu-2020,
    title={Convolutional Neural Networks with Intermediate Loss for 3D Super-Resolution of CT and MRI Scans},
    author={Mariana-Iuliana Georgescu and Radu Tudor Ionescu and Nicolae Verga},
    year={2020}, 
    book={IEEE Access}
}
```
The citation for the paper is found above, with the offical github repository found here: https://github.com/lilygeorgescu/3d-super-res-cnn

### Summary: 
This paper explores using two seperate 2D SR Convolutional Neural Network (CNN) models to accomplish 3D SR. Effectively, if you have a 3D image with axes [x,y,z] and you want to double its size, you first have a model (`CNN_1`) double the height and width of slices along a given axis by treating each slice as a 2D image:

```mermaid
  flowchart LR
  id1["3D img:[x,y,z]"] --> id2["[x,y,0]"] --> id0["CNN_1"] --> id6["[2x,2y,0]"] --> id10["3D img:[2x,2y,z]"] 
  id1["3D img:[x,y,z]"] --> id3["[x,y,1]"] --> id01["CNN_1"] --> id7["[2x,2y,1]"] --> id10["3D img:[2x,2y,z]"]
  id1["3D img:[x,y,z]"] --> id4["..."] --> id02["CNN_1"] --> id8["..."] --> id10["3D img:[2x,2y,z]"]
  id1["3D img:[x,y,z]"] --> id5["[x,y,z]"] --> id03["CNN_1"] --> id9["[2x,2y,z]"] --> id10["3D img:[2x,2y,z]"]
```

Then you use a second model (`CNN_2`) which only doubles the remaining dimension of the 3D image by taking 2D slices from a different orientation:

```mermaid
  flowchart LR
  id1["3D img:[x,y,z]"] --> id2["[0,y,z]"] --> id0["CNN_2"] --> id6["[0,y,2z]"] --> id10["3D img:[x,y,2z]"] 
  id1["3D img:[x,y,z]"] --> id3["[1,y,z]"] --> id01["CNN_2"] --> id7["[1,2y,2z]"] --> id10["3D img:[x,y,2z]"]
  id1["3D img:[x,y,z]"] --> id4["..."] --> id02["CNN_2"] --> id8["..."] --> id10["3D img:[2x,2y,z]"]
  id1["3D img:[x,y,z]"] --> id5["[x,y,z]"] --> id03["CNN_2"] --> id9["[x,y,2z]"] --> id10["3D img:[x,y,2z]"]
```

By combining these two methods, you can effectively do 3D SR using 2D convolutional layers with fewer weights (and thus less training time and resources) than 3D convolutional layers.

```mermaid
  flowchart LR
  id1["3D img:[x,y,z]"] --> id0["CNN_1"] --> id01["CNN_2"] --> id2["3D img:[2x,2y,2z]"]
```

## File list:
```
./main.ipynb : Jupyter notebook used in the creation and training of this SR model type

./CNNIL_save_network1_39.p : Saved weights for the first CNN block

./CNNIL_save_network2_39.p : Saved weights for the second CNN block

```

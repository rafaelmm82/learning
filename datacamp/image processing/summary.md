# Image Processing

This summary group a list of topics and small comments on some of them.
A list of tools, frameworks and related content start the content followed
by the topics splitted in small subjects.

## Resource List

- [scikit-image](http://scikit-image.org)
- [numpy](http://numpy.org)
- [matplotlib](http://matplotlib.org)

## Topics

### Fundamentals

- Purposes of Image Processing
> *Visualization*, catch what are not 'visible'.
> *Biultifying*, sharpening and resotation.
> *Retriving*, seek for images of interesting.
> *Measure*, objets or patterns.
> *Recognition*, distinguish objets witin an image.

- Representation
> Usually using matrix representation, os multilayers matrix representation. Usually [0-255] levels, if grayscale.

- show image (plt.imshow(),plt.show() or skimage.io.imshow(), io.show())

- Channels (Red, Green, Blue), contributions of each, histogram
- Applications of histogram
> Analysis, Thresholding, Brightness and contrast, Equalize.

### Simple operations

- thresholding (simplest way to segmet an image), 1 if >, 0 if <
> works well on high contrast grayscalle images. There two types, global (histogram based) or local (adaptative region based)
- 
import MNIST

using Colors
using Images
using ImageView


"""
    mnist_img(row)
    
    Obtain the row of the MNIST data matrix as image.
"""
function mnist_img(row)
    img = Images.grayim(convert( Images.Image{Gray}, reshape(row, (28,28)) ))
    img["spatialorder"] = ["y", "x"]
    return img
end


"""
    mnist_view(img)
    
    View an image obtained by mnist_img.
"""
mnist_view(img) = ImageView.view(img) # xy=["y","x"])


"""
    mnist_saveview(filename, imgview)
    
    Save the image obtained by mnist_view(example)
"""
mnist_saveimg(filename, img) = Images.save(filename, img)


function drawmnist()
    X, y = MNIST.traindata()

    srand(1337)

    # select first three of each class, normalize and transpose
    X_0 = X[:,(y.==0.0)][:,1:3]' ./ 255.0
    X_1 = X[:,(y.==1.0)][:,1:3]' ./ 255.0
    X_2 = X[:,(y.==2.0)][:,1:3]' ./ 255.0
    X_3 = X[:,(y.==3.0)][:,1:3]' ./ 255.0
    X_4 = X[:,(y.==4.0)][:,1:3]' ./ 255.0
    X_5 = X[:,(y.==5.0)][:,1:3]' ./ 255.0
    X_6 = X[:,(y.==6.0)][:,1:3]' ./ 255.0
    X_7 = X[:,(y.==7.0)][:,1:3]' ./ 255.0
    X_8 = X[:,(y.==8.0)][:,1:3]' ./ 255.0
    X_9 = X[:,(y.==9.0)][:,1:3]' ./ 255.0

    # cleanup
    X = nothing
    y = nothing
    gc()

    mkdir("./img")
    for i in 1:3
        mnist_saveimg("./img/0_"*string(i)*".pdf", mnist_img(X_0[i,:]))
        mnist_saveimg("./img/1_"*string(i)*".pdf", mnist_img(X_1[i,:]))
        mnist_saveimg("./img/2_"*string(i)*".pdf", mnist_img(X_2[i,:]))
        mnist_saveimg("./img/3_"*string(i)*".pdf", mnist_img(X_3[i,:]))
        mnist_saveimg("./img/4_"*string(i)*".pdf", mnist_img(X_4[i,:]))
        mnist_saveimg("./img/5_"*string(i)*".pdf", mnist_img(X_5[i,:]))
        mnist_saveimg("./img/6_"*string(i)*".pdf", mnist_img(X_6[i,:]))
        mnist_saveimg("./img/7_"*string(i)*".pdf", mnist_img(X_7[i,:]))
        mnist_saveimg("./img/8_"*string(i)*".pdf", mnist_img(X_8[i,:]))
        mnist_saveimg("./img/9_"*string(i)*".pdf", mnist_img(X_9[i,:]))
    end

end

drawmnist()
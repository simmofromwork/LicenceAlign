#Pyscript environment to be inserted in html header.
<py-config>
        - autoclose_loader: true
        - runtimes:
          - src: "https://cdn.jsdelivr.net/pyodide/dev/full/pyodide.js"
            name: pyodide-dev
            lang: python
      </py-config>
    <py-env>
        - Pillow
        - opencv-python
        - numpy
        - paths: 
            - /walicence.png
    </py-env>


#Actual pyscript that sits inside a section tag in the html file. 
#Alternatively have the .py file as a resource in your div tag however I had some trouble with this on my hosting service so I nested the code in the html file.


<div class="card flex flex-col md:max-w-4xl md:flex-col">
                <h3 class="text-3xl font-bold mb-5 md:text-4xl">Client side Computer Vision Licence Extraction</h3>
                        <div class="card-image">
                            <a>
                                <img src="img/browhat.jpg" alt="Card Image">
                            </a>
                            <br>
                            <p class="text-xs">This is an example before and after of document alignment</p>
                            <br>

                        </div>

                    <div id="output_upload_pillow" class="flex flex-col w-full m-auto"></div>
                    <label for="Upload a PNG image"></label><input type="file" id="file-upload-pillow">
                    <py-script >#!/home/dh_tp7g6m/opt/python-3.11.4/bin/python3

from js import document, console, Uint8Array, window, File
import asyncio
import io
import os
import numpy
from pyodide.ffi import create_proxy
import cv2 
from PIL import Image, ImageFilter
from pyodide.http import pyfetch

async def _upload_change_and_show(e):
 #Get the first file from upload
    file_list = e.target.files
    first_item = file_list.item(0)

    #Get the data from the files arrayBuffer as an array of unsigned bytes
    array_buf = Uint8Array.new(await first_item.arrayBuffer())

    #BytesIO wants a bytes-like object, so convert to bytearray first
    bytes_list = bytearray(array_buf)
    my_bytes = io.BytesIO(bytes_list) 

    #Create PIL image from np array
    my_image = Image.open(my_bytes)
    console.log(f"{my_image.format= } {my_image.width= } {my_image.height= }")
    my_image = numpy.array(my_image) 
    # Convert RGB to BGR 
    my_image = my_image[:, :, ::-1].copy()
    console.log("before template")
    #get template image.
    temp = await get_template()
    console.log("after temp return")
    #align
    my_image = await find_document_edges_and_align(my_image, temp , 0.1)
    #convert back to PIL
    console.log("After calling align in upload")
    color_converted = cv2.cvtColor(my_image, cv2.COLOR_BGR2RGB)
    my_image = Image.fromarray(color_converted)
    #Convert Pillow object array back into File type that createObjectURL will take
    my_stream = io.BytesIO()
    my_image.save(my_stream, format="PNG")

    #Create a JS File object with our data and the proper mime type
    image_file = File.new([Uint8Array.new(my_stream.getvalue())], "new_image_file.png", {type: "image/png"})

    #Create new tag and insert into page
    #First clear the previous image in the div tag if its there.
    document.getElementById("output_upload_pillow").innerHTML = ""
    new_image = document.createElement('img')
    new_image.src = window.URL.createObjectURL(image_file)
    document.getElementById("output_upload_pillow").appendChild(new_image)

# Run image processing code above whenever file is uploaded    
upload_file = create_proxy(_upload_change_and_show)
document.getElementById("file-upload-pillow").addEventListener("change", upload_file)

async def get_template():
    temp = numpy.array(Image.open('walicence.png'))
    return temp



#This function aligns and crops the image

async def find_document_edges_and_align(cv_image, template, keepPercent):
    # Read the input image
    maxFeatures = 4000
    img = cv_image
    console.log("ABove gray conversion in align")
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    templateGray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    console.log("Below gray conversion in align")
    # Apply GaussianBlur to reduce noise and help edge detection
    blurred = cv2.GaussianBlur(gray, (3, 3), 0 )
    temp_blur = cv2.GaussianBlur(templateGray, (3,3), 0)
    # Detetct orb features

    orb = cv2.ORB_create(maxFeatures)
    (kpsA, descsA) = orb.detectAndCompute(blurred, None)
    (kpsB, descsB) = orb.detectAndCompute(temp_blur, None)


    #match features
    method = cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING
    matcher = cv2.DescriptorMatcher_create(method)
    matches = matcher.match(descsA, descsB, None)

    # sort the matches by their distance (the smaller the distance,
    # the "more similar" the features are)
    matches = sorted(matches, key=lambda x: x.distance)
    # keep only the top matches
    keep = int(len(matches) * keepPercent)
    matches = matches[:keep]
    # check to see if we should visualize the matched keypoints

    matchedVis = cv2.drawMatches(img, kpsA, template, kpsB,
                                     matches, None)
    
    # allocate memory for the keypoints (x, y)-coordinates from the
    # top matches -- we'll use these coordinates to compute our
    # homography matrix
    ptsA = numpy.zeros((len(matches), 2), dtype="float")
    ptsB = numpy.zeros((len(matches), 2), dtype="float")
    # loop over the top matches
    for (i, m) in enumerate(matches):
        # indicate that the two keypoints in the respective images
        # map to each other
        ptsA[i] = kpsA[m.queryIdx].pt
        ptsB[i] = kpsB[m.trainIdx].pt

    # compute the homography matrix between the two sets of matched
    # points
    (H, mask) = cv2.findHomography(ptsA, ptsB, method=cv2.RANSAC)
    # use the homography matrix to align the images
    (h, w) = template.shape[:2]
    aligned = cv2.warpPerspective(img, H, (w, h))
    # return the aligned image
    console.log("returning aligned")
    return aligned
    </py-script>

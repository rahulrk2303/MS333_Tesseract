from __future__ import print_function
import binascii
import struct
# from PIL import Image
import numpy as np
import scipy
import scipy.misc
import scipy.cluster
import webcolors



NUM_CLUSTERS = 3

def closest_colour(requested_colour):
	min_colours = {}
 #    color_list = {'#000000': 'black', '#0000ff': 'blue', '#00ff00': 'green', 
	# '#ff0000': 'red', '#ffffff': 'white', '#ffff00': 'yellow', '#000080': 'navy'} 

	color_list = {
		'#000000': 'Rental-Black',
		'#0000ff': 'Foreign Embassy-Blue', 
		'#00ff00': 'Electric-Green', 
		'#ff0000': 'President/Governors-Red', 
		'#ffffff': 'Private-White', 
		'#ffff00': 'Commercial-Yellow'
	 }
	
	# for key, name in webcolors.html4_hex_to_names.items():
	for key, name in color_list.items():
		r_c, g_c, b_c = webcolors.hex_to_rgb(key)
		rd = (r_c - requested_colour[0]) ** 2
		gd = (g_c - requested_colour[1]) ** 2
		bd = (b_c - requested_colour[2]) ** 2
		min_colours[(rd + gd + bd)] = name
	return min_colours[min(min_colours.keys())]

def get_colour_name(requested_colour):
	try:
		closest_name = actual_name = webcolors.rgb_to_name(requested_colour)
	except ValueError:
		closest_name = closest_colour(requested_colour)
		actual_name = None
	return actual_name, closest_name

def process_lp_color(im):

	ar = np.asarray(im)
	shape = ar.shape
	ar = ar.reshape(scipy.product(shape[:2]), shape[2]).astype(float)

	print('finding clusters')
	codes, dist = scipy.cluster.vq.kmeans(ar, NUM_CLUSTERS)
	print('cluster centres:\n', codes)

	vecs, dist = scipy.cluster.vq.vq(ar, codes)         # assign codes
	counts, bins = scipy.histogram(vecs, len(codes))    # count occurrences

	index_max = scipy.argmax(counts)                    # find most frequent
	peak = codes[index_max]
	peak = [int(i) for i in peak]
	colour = binascii.hexlify(bytearray(int(c) for c in peak)).decode('ascii')
	print('most frequent is %s (#%s)' % (peak, colour))

	# plt.imshow([[peak]])
	# plt.show()
	# print(peak)
	actual_name, closest_name = get_colour_name(tuple(peak))

	print ("Colour :", closest_name)
	return closest_name
			
	# if final_colors:
	# 	max_key = max(final_colors, key=final_colors.get)
	# 	return (max_key.__name__)
	# else:
	# 	return ''
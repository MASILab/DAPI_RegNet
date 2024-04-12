#!/bin/bash
DATA_DIR="/nfs2/baos1/rudravg"
RESULTS_DIR="${DATA_DIR}/DeepCell_Results"

# Iterate over all files in the DATA_DIR directory
find "${DATA_DIR}" -maxdepth 1 -type f | while read FILE; do
    # Extract the filename from the full path
        name=$(basename "$FILE")
	   echo $name 
	   # Check if the file exists in the RESULTS_DIR directory
	        if [ ! -f "${RESULTS_DIR}/${name}" ]; then
			        # If the file does not exist, process it with the Docker command
				        
				        # Create a relative path for the file inside the Docker container
					        RELATIVE_PATH=${FILE#${DATA_DIR}/}  # Remove the DATA_DIR part from FILE path
						        MOUNT_DIR="/data"
							        APPLICATION="mesmer"
								        
								        # Run the Docker command
									        docker run --gpus 1 \
											            -v "${DATA_DIR}:${MOUNT_DIR}" \
												                vanvalenlab/deepcell-applications:latest-gpu \
														            $APPLICATION \
															                --nuclear-image "${MOUNT_DIR}/${RELATIVE_PATH}" \
																	            --output-directory "${MOUNT_DIR}/DeepCell_Results" \
																		                --output-name "${name}" \
																				            --compartment "nuclear" \
																					                --image-mpp 0.648
										    fi
									    done


This was a task by cellula's technologies and all data rights are reserved to their company,
I learned how to work with computer vision (PyTorch) and how to analyze/segment videos frame by frame
I used GRU and trained the model to analyse quickly
then launched the model using Flask.

Libraries used:

Core Deep Learning:
| Library         | Purpose                                                          |
| --------------- | ---------------------------------------------------------------- |
| **torch**       | Main PyTorch framework for deep learning.                        |
| **torchvision** | Provides pretrained models (MobileNetV3), image transforms, etc. |
| **numpy**       | Numerical operations, array handling, frame sampling, etc.       |

Computer Vision / Video Handling:
| Library                 | Purpose                                                             |
| ----------------------- | ------------------------------------------------------------------- |
| **opencv-python (cv2)** | Video reading (frame extraction), resizing, and RGB/BGR conversion. |

Web Framework (Backend):
| Library      | Purpose                                                            |
| ------------ | ------------------------------------------------------------------ |
| **Flask**    | Lightweight web framework serving the model as an API/web app.     |
| **Werkzeug** | Used internally by Flask for request handling, file uploads, etc.  |
| **jinja2**   | Templating engine for rendering HTML (`index.html`, `about.html`). |

Utilities:
| Library                               | Purpose                                               |
| ------------------------------------- | ----------------------------------------------------- |
| **json**                              | For loading the class labels from `class_names.json`. |
| **os / pathlib**                      | File and directory management.                        |
| **logging**                           | Output and debugging info in the terminal.            |
| **numpy / cv2 / torch / torchvision** | Data preprocessing, inference, and tensor handling.   |

Development:
| Library                              | Purpose                                                                |
| ------------------------------------ | ---------------------------------------------------------------------- |
| **Flask-CORS** *(optional)*          | If you later allow requests from other domains (e.g., front-end apps). |
| **gunicorn / waitress** *(optional)* | For deploying Flask app in production (instead of `flask run`).        |

Command to install all of the used libraries if you want to run the code yourself:

pip install torch torchvision flask opencv-python numpy

When in need to test/run the application open the folder and type in the search CMD
then type

python app.py

no need to re-train the model since it already exists

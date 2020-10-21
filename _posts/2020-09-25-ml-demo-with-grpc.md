---
layout: single
title:  "ML/DL demo pipeline with gRPC. Wait. What?"
excerpt: "In this blog post we will talk about ML/DL model demo pipeline using gRPC framework"
categories: tutorial
tags: [pipeline, demo, machine learning, deep learning, ML ops, python, REST, gRPC, tutorial]

header:
    teaser: https://grpc.io/img/landing-2.svg
    og_image: https://grpc.io/img/landing-2.svg
---

![front image](https://grpc.io/img/landing-2.svg)

{% include toc title="Table of Contents" %}

# ML/DL model demo pipeline with gRPC. Wait. What?

Wait. What? gRPC. Why?

One day a customer asked to deploy the model we designed using gRPC. As being familiar with REST, the use of gRPC seemed excessive. But once you understand how to work with gRPC and Protocol Buffers you speed up receiving and sending data significantly which in turn speed up inference speed.

In this post, I'm not going to describe what gRPC is. There are many articles in which authors compare JSON/REST with gRPC/Protobuf. This post is about using the latest in a typical computer vision task. 

Deep Learning and Computer Vision related topics won't be brought up.
{: .notice--info}

To easily follow along with this tutorial, please checkout out the source code on my [GitHub profile]().
{: .notice--info}

## Interface with Pydantic & Protobuf

Protocol Buffers require to generalize and use strict types for Input _(Request)_ and Output _(Response)_ data. Actually, that can be useful as well to make sure that fields match up exactly. Let's start with Interface for an Emotion Recognition CV task using Pydantic for Python and Protobuf for gRPC. I'm going to describe it all together to make it easier to understand.

### Bounding Box

In CV-based tasks a detected usually is represented by a Bounding Box coordinates. So let's specify BBox interface in Python code and Protobuf file.

First, let's specify BBox object in Python code. We need two points to represent bounding box. For example - _(x1, y1, x2, y2)_ as shown in the example below. Also, we specify that each variable is float, and let's assign zero as the default value.

```python
class BBox(pydantic.BaseModel):
    x1: float = 0
    y1: float = 0
    x2: float = 0
    y2: float = 0
```

Now, we have to specify the same object in Protobuf `.proto` file for gRPC connection. Visit [Protocol Buffers](https://developers.google.com/protocol-buffers) for more information about syntax and available types.

```editorconfig
message BBox {
  float x1 = 1;
  float y1 = 2;
  float x2 = 3;
  float y2 = 4;
}
```

_1,2,3,4_ are not default values here. It is a unique number to identify fields in the binary message format. 


### Expression

Then, if we are talking about the Emotion Recognition task we need to specify Emotion. Right? Let's do the same with pydantic and protobuf

```python
class Expression(pydantic.BaseModel):
    neutral: float = 0
    anger: float = 0
    disgust: float = 0
    fear: float = 0
    happiness: float = 0
    sadness: float = 0
    surprise: float = 0
```
```editorconfig
message Expression {
  float neutral = 1;
  float anger = 2;
  float disgust = 3;
  float fear = 4;
  float happiness = 5;
  float sadness = 6;
  float surprise = 7;
}
```

### Result

Now we want to specify model result. And one more time, it's easy to do.
```python
class Result(pydantic.BaseModel):
    bbox: BBox = BBox()
    expression: Expression = Expression()
```

```editorconfig
message Result {
  BBox bbox = 1;
  Expression expression = 2;
}
```

Yes, we can use custom types we defined before in previous sections.

### Request / Response

The last we need to do is to specify Request and Response. For request let's use binary image data and for a response a list of results that we defined in the previous section. 

```editorconfig
message Request {
  bytes image = 1;
}
message Response {
  repeated Result response = 1;
}
```

So now let's also define an Interface for our model. 

```python
class Response(pydantic.BaseModel):
    response: List[Result]

class BaseModel(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def process_sample(self, image: Image) -> Response:
        raise NotImplementedError
```

So, for our model Image is the input and Response is the output. Now we can use any model extended from BaseModel and overridden `process_sample`. 

That's it. Our API is ready for both sides - Python and Protocol Buffers. Pydantic usage doesn't allow us to use other types so a response will always match protobuf format.


## Model

Now, when everything is set up, we can make our super CV model. Ready? Let's do it!

```python
class MockModel(BaseModel):
    def process_sample(self, image: Image) -> Response:
        image = image.resize((128, 128))
        return Response(response=[Result()])
```

No kidding. As I said this post is not about CV, it's about ML ops - providing demo to your customer. 

So what we've done here. We extended our interface BaseModel with `process_sample` that takes `Image` as input and return `Response` as an output which is important as gRPC expects to have a strong typed Protocol Buffer which the interfaces match.

## Pipeline

RPC is a Remote Protocol Call, so now we need to define a function that can be called. To do this, it's necessary to define a service with that function in `proto` file.

```editorconfig
service Pipeline {
  rpc run_pipeline (Request) returns (Response) {}
}
```

Here we have a service named `Pipeline` with a function `run_pipeline` that takes as input our previously defined `Request` and as output `Response`. That's almost it. 

To that end, we need to update gRPC code used by our application. The command below will generate 2 `.py` files.

```commandline
python -m grpc_tools.protoc -I../../protos --python_out=. --grpc_python_out=. ../../protos/helloworld.proto
```

The last thing we have to do is to make a server application that processes gRPC connections.


```python
# Don't forget to import generated protobuf python files
import protobuf_pb2
import protobuf_pb2_grpc

# Create our model
model = MockModel()

# Define a class extended from PipelineServicer
# Pipeline is a service name we defined previously
class Pipeline(protobuf_pb2_grpc.PipelineServicer):
    # We need to override run_pipeline as we declare this function in proto file
    def run_pipeline(self, request, context):
        # Create an image from binary
        image = Image.open(io.BytesIO(request.image))
        # Call process_sample we overrided in MockModel()
        response = model.process_sample(image).dict()
        # We know that it returns Response so now we can reply
        return protobuf_pb2.Response(response=response["response"])
```

Also, we need to define a server itself, it can be done this way. For more information about gRPC, please, visit its official [site](https://grpc.io/docs/).

```python
def serve():
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=multiprocessing.cpu_count())
    )
    # Adding our Pipeline to server
    protobuf_pb2_grpc.add_PipelineServicer_to_server(Pipeline(), server)
    server.add_insecure_port(HOST)
    server.start()
    server.wait_for_termination()

if __name__ == "__main__":
    serve()
```

In order to test we can create a simple Client application with gRPC Stub

```python
def run(image_path):
    with grpc.insecure_channel(HOST) as channel:
        # create a stub with which one connection to the server can be established    
        stub = protobuf_pb2_grpc.PipelineStub(channel)
        binary = open(image_path, "rb").read()
        # call run_pipeline function using stub
        response = stub.run_pipeline(protobuf_pb2.Request(image=binary))
        channel.close()


if __name__ == "__main__":
    run("harold.jpg")
```


## Conclusion

In this blog post, we've learned how to use gRPC for simple model deployment. As it can be clearly gRPC is a solution to consider for providing ML/DL based solutions for customers. 
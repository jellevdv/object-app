import { cameraWithTensors } from '@tensorflow/tfjs-react-native';
import { Camera } from 'expo-camera';
import React, { useState, useRef, useEffect } from 'react';
import { Dimensions, LogBox, Platform, View, StyleSheet } from 'react-native';
import Canvas from 'react-native-canvas';
import * as cocoSsd from '@tensorflow-models/coco-ssd';
import * as tf from '@tensorflow/tfjs';

const TensorCamera = cameraWithTensors(Camera);
const { width, height } = Dimensions.get('window');

LogBox.ignoreAllLogs(true);
export default function App() {
  const [model, setModel] = useState<cocoSsd.ObjectDetection>();
  let context = useRef<CanvasRenderingContext2D>();
  let canvas = useRef<Canvas>();

  let textureDims =
    Platform.OS == 'ios'
      ? { height: 1920, width: 1080 }
      : { height: 1200, width: 1600 };

  function handleCameraStream(images: any) {
    const loop = async () => {
      const nextImageTensor = images.next().value;
      console.log(nextImageTensor);

      if (!model || !nextImageTensor)
        throw new Error('No model or image tensor');

      model
        .detect(nextImageTensor)
        .then((predicition: any) => {
          drawRectangle(predicition, nextImageTensor);
        })
        .catch((e: any) => {
          console.log(e);
        });

      requestAnimationFrame(loop);
    };
    loop();
  }

  function drawRectangle(
    predicitions: cocoSsd.DetectedObject[],
    nextImageTensor: any
  ) {
    if (!context.current || !canvas.current) return;

    //match size of camera preview
    const scaleWidth = width / nextImageTensor.shape[1];
    const scaleHeight = height / nextImageTensor.shape[0];

    const flipHorizontal = Platform.OS == 'ios' ? false : true;

    context.current.clearRect(0, 0, width, height);

    for (const predicition of predicitions) {
      const [x, y, width, height] = predicition.bbox;

      const boundingBoxX = flipHorizontal
        ? canvas.current.width - x * scaleWidth - width * scaleWidth
        : x * scaleWidth;
      const boundingBoxY = y * scaleHeight;

      context.current.strokeRect(
        boundingBoxX,
        boundingBoxY,
        width * scaleWidth,
        height * scaleHeight
      );

      context.current.strokeText(
        predicition.class,
        boundingBoxX - 5,
        boundingBoxY - 5
      );
    }
  }

  async function handleCanvas(can: Canvas) {
    if (can) {
      can.width = width;
      can.height = height;
      //todo fix this
      let ctx: CanvasRenderingContext2D = can.getContext('2d');
      ctx.strokeStyle = 'red';
      ctx.fillStyle = 'red';
      ctx.lineWidth = 1;

      context.current = ctx;
      canvas.current = can;
    }
  }

  useEffect(() => {
    (async () => {
      const { status } = await Camera.requestCameraPermissionsAsync();
      await tf.ready();
      setModel(await cocoSsd.load());
    })();
  }, []);

  return (
    <View style={styles.container}>
      <TensorCamera
        style={styles.camera}
        type={Camera.Constants.Type.back}
        cameraTextureHeight={textureDims.height}
        cameraTextureWidth={textureDims.width}
        resizeHeight={200}
        resizeWidth={152}
        resizeDepth={3}
        onReady={handleCameraStream}
        autorender={true}
        useCustomShadersToResize={false}
      />
      <Canvas style={styles.canvas} ref={handleCanvas} />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
  },
  camera: {
    width: '100%',
    height: '100%',
  },
  canvas: {
    position: 'absolute',
    zIndex: 1000000,
    width: '100%',
    height: '100%',
  },
});

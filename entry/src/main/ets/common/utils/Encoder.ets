import util from '@ohos.util';
import resourceManager from '@ohos.resourceManager';
import Logger from './Logger'

export class Encoder {

  public async imageEncode(imageName: string) {
    let imageFinalCode : string = "";
    try {
      let imageBase64Buffer = await getContext().resourceManager.getMediaBase64ByName(imageName);
      // 去除编码头
      imageBase64Buffer = imageBase64Buffer.replace(/^data:image\/(png|jpg|bmp);base64,/, '')

      // imageBase64Buffer = imageBase64Buffer;

      // URL编码
      imageFinalCode = `${encodeURIComponent(imageBase64Buffer)}`;

      Logger.info("encode media to base64 successfully, imageFinalCode:", imageFinalCode.slice(15));
    } catch(err) {
      Logger.error("get Media base64 by name failed, imageName:", JSON.stringify(err) + ';       ' + imageName);
    }
    return imageFinalCode
  }


}
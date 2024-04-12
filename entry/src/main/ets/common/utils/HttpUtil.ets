import http from '@ohos.net.http';
import { ImageItem } from '../../viewmodel/ImageItem'
import { GeneralResponseUnit } from '../../viewmodel/ResponseUnitModel'
import { CommonConstants } from '../../common/constants/CommonConstants'
import { Encoder } from './Encoder'
import { ACCESS_TOKEN } from '../../common/constants/Token'
import router from '@ohos.router';
import Logger from '../../common/utils/Logger'

export class HttpHelper {
  private imageName: string;
  private encoder: Encoder = new Encoder();

  constructor(imageName: string) {
    this.imageName = imageName;
  }

  /**
   * 将给定的图片上传到百度云, 并返回通用识别列表
   * @returns
   */
  async generalImageRequest(): Promise<Array<GeneralResponseUnit>> {

    let imageCode = await this.encoder.imageEncode(this.imageName)


    let httpRequest = http.createHttp();
    let url = CommonConstants.BAIDU_GENERAL_URL + '?' + 'access_token=' + ACCESS_TOKEN;
    let httpResponse = httpRequest.request(url, {
      method: http.RequestMethod.POST,
      // readTimeout: CommonConstants.REQUEST_TIMEOUT,
      header: {
        "Content-Type": "application/x-www-form-urlencoded"
      },
      // connectTimeout: CommonConstants.REQUEST_TIMEOUT,
      extraData: {
        "image": imageCode,
        "baidu_num": CommonConstants.BAIKE_NUM
      }
    });

    let results: Array<GeneralResponseUnit> = [];
    return httpResponse.then((response: http.HttpResponse) => {
      if (response.responseCode == CommonConstants.HTTP_STATUS_CODE_OK) {
        let data = JSON.parse(response.result as string); // json格式的字符串
        let result = data.result; // Object数组
        result.forEach((item, index) => {
          let resultUnit = new GeneralResponseUnit(item.keyword, item.root, item.score, item.baike_info.description);
          results.push(resultUnit);
        })
        // DEBUG
        Logger.info("get response successfully", JSON.stringify(response))
      }
      else {
        // DEBUG
        Logger.error("get response failed", JSON.stringify(response))
      }
      return results;
    })
      .catch(( error ) => {
        // DEBUG
        Logger.error("request error", JSON.stringify(error)); /**********************************************************************BUGBUGBUG is here !!!!!!!!!!!!!*******************************************************/
        return results;
      })
  }
}


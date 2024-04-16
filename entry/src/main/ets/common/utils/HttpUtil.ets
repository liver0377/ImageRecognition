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
    let options = {
      method: http.RequestMethod.POST,
      readTimeout: CommonConstants.HTTP_READ_TIMEOUT,
      connectTimeout: CommonConstants.REQUEST_TIMEOUT,
      header: {
        "Content-Type": "application/x-www-form-urlencoded"
      },
      extraData: `image=${imageCode}&baike_num=${CommonConstants.BAIKE_NUM}`
    }

    // let test_url = "http://172.16.57.196:9588/news/getNewsType"
    // let test_options = {
    //   method: http.RequestMethod.GET,
    //   readTimeout: CommonConstants.HTTP_READ_TIMEOUT,
    //   connectTimeout: CommonConstants.HTTP_READ_TIMEOUT,
    //   header: {
    //     "Content-Type": "application/json"
    //   },
    //   extraData: {}
    // }

    let results: Array<GeneralResponseUnit> = [];
    let httpResponseResult = httpRequest.request(url, options);

    return httpResponseResult.then((response: http.HttpResponse) => {
      if (response.responseCode == CommonConstants.HTTP_STATUS_CODE_OK) {
        let data = JSON.parse(response.result as string); // json格式的字符串
        Logger.info("response.data: ", JSON.stringify(data))
        let result = data.result; // Object数组
         result.forEach((item, index) => {
           let resultUnit = new GeneralResponseUnit(item.keyword, item.root, item.score, item.baike_info == undefined ? "" : item.baike_info.description);
           results.push(resultUnit);
         })
      }
      else {
        Logger.error("response code is ", response.responseCode.toString())
      }
      Logger.info("request success", "");
      return results;
    })
      .catch(( error ) => {

        Logger.error("request error", JSON.stringify(error));
        return results;
      })
  }
}


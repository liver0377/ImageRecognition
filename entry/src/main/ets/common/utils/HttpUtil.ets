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


    let results: Array<GeneralResponseUnit> = [];
    let httpResponseResult = httpRequest.request(url, options);

    return httpResponseResult.then((response: http.HttpResponse) => {
      if (response.responseCode == CommonConstants.HTTP_STATUS_CODE_OK) {
        let data = JSON.parse(response.result as string); // json格式的字符串
        Logger.info("response.data: ", JSON.stringify(data))
        let result = data.result; // Object数组
         result.forEach((item, index) => {
           let baike_description :string = "";
           let baike_url: string = "";
           let image_url:string = "";

           if (item.baike_info != undefined) {
               baike_description= item.baike_info.descrition;
               baike_url = item.baike_info.baike_url;
               image_url = item.baike_info.image_url;
           }
           let resultUnit = new GeneralResponseUnit(item.keyword, item.root, item.score, baike_description, image_url, baike_url);
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


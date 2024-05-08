import http from '@ohos.net.http';
import { GeneralResponseUnit } from '../../viewmodel/ResponseUnitModel'
import { CommonConstants } from '../../common/constants/CommonConstants'
import { Encoder } from './Encoder'
import { ACCESS_TOKEN } from '../../common/constants/Token'
import Logger from '../../common/utils/Logger'

export class HttpHelper {
  private imageName: string;
  private encoder: Encoder = new Encoder();

  constructor(imageName: string) {
    this.imageName = imageName;
  }

  async imageRequestHelper(type: string) {
    let imageCode = await this.encoder.imageEncode(this.imageName)
    let httpRequest = http.createHttp();
    let prefix: string;
    let extraData: string;

    if (type == "general") {
      // prefix = CommonConstants.BAIDU_GENERAL_URL;
      prefix = CommonConstants.MY_GENERAL_URL
      extraData = `image=${imageCode}&baike_num=${CommonConstants.BAIKE_NUM}`;
    }
    if (type == "animal") {
      prefix = CommonConstants.BAIDU_ANIMAL_URL;
      extraData = `image=${imageCode}&top_num=${CommonConstants.BAIKE_NUM}&baike_num=${CommonConstants.BAIKE_NUM}`;
    }
    if (type == "plant") {
      prefix = CommonConstants.BAIDU_PLANT_URL;
      extraData = `image=${imageCode}&top_num=${CommonConstants.BAIKE_NUM}&baike_num=${CommonConstants.BAIKE_NUM}`;
    }
    if (type == "logo") {
      prefix = CommonConstants.BAIDU_LOGO_URL;
      extraData = `image='${imageCode}`
    }

    // todo
    if (type == "money") {
      prefix = CommonConstants.BAIDU_MONEY_URL;
      extraData = `image=${imageCode}`
    }

    let url = prefix + '?' + 'access_token=' + ACCESS_TOKEN;

    let options = {
      method: http.RequestMethod.POST,
       readTimeout: CommonConstants.HTTP_READ_TIMEOUT,
       connectTimeout: CommonConstants.REQUEST_TIMEOUT,
      header: {
        "Content-Type": "application/x-www-form-urlencoded"
      },
      extraData: extraData
    }

    let results: Array<GeneralResponseUnit> = [];
    let httpResponseResult = httpRequest.request(url, options);

    return httpResponseResult.then((response: http.HttpResponse) => {
      if (response.responseCode == CommonConstants.HTTP_STATUS_CODE_OK) {
        let data = JSON.parse(response.result as string); // json格式的字符串
        Logger.info("response.data: ", JSON.stringify(data))
        let result = data.result; // Object数组
        result.forEach((item) => {
          let keyword: string = "";
          let tag: string = "";
          let score: number = 0;
          let description: string = item.baike_info == undefined ? "xxx" : item.baike_info.description;
          let imageUrl: string = item.baike_info == undefined ? "" : item.baike_info.image_url;
          let baikeUrl: string = item.baike_info == undefined ? "" : item.baike_info.baike_url;

          if (type == "general") {
            keyword = item.keyword;
            tag = item.root;
            score = item.score;
          }

          if (type == "animal") {
            keyword = item.name;
            tag = "动物"
            score = item.score
          }

          if (type == "plant") {
            keyword = item.name;
            tag = "植物"
            score = item.score
          }

          if (type == "logo") {
            keyword = item.name;
            tag = "logo"
            score = item.probability
          }

          if (type == "money") {
            keyword = item.currencyName;
            tag = "货币"
            score = null;
          }

          let resultUnit = new GeneralResponseUnit(keyword, tag, score, description, imageUrl, baikeUrl);
          results.push(resultUnit);
        })
      }
      else {
        Logger.error("response code is ", response.responseCode.toString())
      }
      Logger.info("request success", "");
      return results;
    })
      .catch((error) => {

        Logger.error("request error", JSON.stringify(error));
        return results;
      })
  }
  /**
   * 将给定的图片上传到百度云, 并返回通用识别列表
   * @returns
   */
  async generalImageRequest(): Promise<Array<GeneralResponseUnit>> {

    return this.imageRequestHelper("general");
  }

  async animalImageRequest(): Promise<Array<GeneralResponseUnit>> {
    return this.imageRequestHelper("animal");
  }

  async plantImageRequest(): Promise<Array<GeneralResponseUnit>> {
    return this.imageRequestHelper("plant")
  }

  async logoImageRequest() : Promise<Array<GeneralResponseUnit>> {
    return this.imageRequestHelper("logo");
  }

  async moneyImageRequest() : Promise<Array<GeneralResponseUnit>> {
    return this.imageRequestHelper("money")
  }
}





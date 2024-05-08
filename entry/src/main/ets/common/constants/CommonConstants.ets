import { ImageItem } from '../../viewmodel/ImageItem'
import resourceManager from '@ohos.resourceManager';
/**
 * Common constants for all features.
 */
export class CommonConstants {
  /**
   * Full percent.
   */
  static readonly FULL_PERCENT: string = '100%';
  static readonly TEN_PERCENT: string = '10%';
  static readonly TWENTY_PERCENT: string = '20%';
  static readonly THIRTY_PERCENT: string = '30%';
  static readonly FORTY_PERCENT: string = '40%';
  static readonly FIFTY_PERCENT: string = '50';
  static readonly SIXTY_PERCENT: string = '60%';
  static readonly SEVENTY_PERCENT: string = '70%';
  static readonly EIGHTY_PERCENT: string = '80%';
  static readonly NINETY_PERCENT: string = '90%';
  static readonly ONE_HUNDRED_PERCENT: string = '100%';

  /**
   * Width-height ratio.
   */
  static readonly ASPECT_RATIO: number = 1;
  /**
   * One hundred.
   */
  static readonly ONE_HUNDRED: number = 100;
  /**
   * A thousand.
   */
  static readonly A_THOUSAND: number = 1000;

  static readonly TIME_UNIT: number = 60;
  /**
   * Initial Time UNIT.
   */
  static readonly INITIAL_TIME_UNIT: string = '00';
  /**
   * Zero padding, 2 bits.
   */
  static readonly PADDING_LENGTH: number = 2;
  /**
   * String zero padding.
   */
  static readonly PADDING_STR: string = '0';
  /**
   * Breath screen status.
   */
  static readonly SCREEN_OFF: string = 'usual.event.SCREEN_OFF';
  /**
   * Operation status of video player 4.
   */
  static readonly OPERATE_STATE: Array<string> = ['prepared', 'playing', 'paused', 'completed'];

  /**
   * image recognition result page
   */
  static readonly RESULT_PAGE: string =  "pages/ResultPage";

  /**
   * baidu image recognition url
   */
  static readonly BAIDU_GENERAL_URL : string = "https://aip.baidubce.com/rest/2.0/image-classify/v2/advanced_general"
  static readonly BAIDU_ANIMAL_URL : string = "https://aip.baidubce.com/rest/2.0/image-classify/v1/animal"
  static readonly BAIDU_PLANT_URL : string = "https://aip.baidubce.com/rest/2.0/image-classify/v1/plant"
  static readonly BAIDU_LOGO_URL : string = "https://aip.baidubce.com/rest/2.0/image-classify/v2/logo"
  static readonly BAIDU_MONEY_URL : string = "https://aip.baidubce.com/rest/2.0/image-classify/v1/currency"

  static readonly MY_GENERAL_URL: string = "http://192.168.100.169:9588/image/"

  /**
   * baidu baike numbers
   */
  static readonly  BAIKE_NUM : number = 5;


  static readonly HTTP_STATUS_CODE_OK : number = 200;
  /**
   *  http request timeout duration
   */
  static readonly REQUEST_TIMEOUT: number = 60000;

  static readonly HTTP_READ_TIMEOUT: number = 60000;


}


/**
 * Image object collection
 */
export const IMAGE_DATA: ImageItem[] = [
  new ImageItem('image_1', "jpg"),
  new ImageItem('image_2', "jpg"),
  new ImageItem('image_3', "jpg"),
  new ImageItem('image_4', "jpg"),
  new ImageItem('image_5', "jpg"),
  new ImageItem('image_6', "jpg"),
  new ImageItem('image_7', "jpg"),
  new ImageItem('image_8', "jpg"),
  new ImageItem('image_9', "jpg"),
  new ImageItem('image_10', "jpg"),
  new ImageItem('image_11', "jpg"),
]



export enum RecognitionType {
  General = "general",
  Animal = "animal",
  Botany = "botany",
  Logo = "logo",
  Money = "money"
}




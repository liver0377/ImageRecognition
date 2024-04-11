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
  static readonly PAGE: string =  "pages/ResultPage";
}


/**
 * Image object collection
 */
export const IMAGE_DATA: ImageItem[] = [
  new ImageItem('image1.jpg', {} as resourceManager.RawFileDescriptor),
  new ImageItem('image2.jpg', {} as resourceManager.RawFileDescriptor),
  new ImageItem('image3.jpg', {} as resourceManager.RawFileDescriptor),
]

export const recognitionVariety : Resource[] = [
  $r('app.string.animal_recognition'),
  $r('app.string.botany_recognition'),
  $r('app.string.logo_recognition'),
  $r('app.string.vegetable_recognition'),
  $r('app.string.money_recognition'),
]

export enum RecognitionType {
  General = "general",
  Animal = "animal",
  Botany = "botany",
  Logo = "logo",
  Vegetable = "vegetable",
  Money = "money"
}



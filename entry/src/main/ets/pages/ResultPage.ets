import { ImageItem } from '../viewmodel/ImageItem'
import { ResultItem } from '../viewmodel/ResultItem'
import { CommonConstants,  RecognitionType } from '../common/constants/CommonConstants'
import { PageConstants } from '../common/constants/PageConstants'
import { ResultTabContent } from '../view/ResultTabContent'
import resourceManager from '@ohos.resourceManager';
import router from '@ohos.router';

@Entry
@Component
struct ResultPage {
  private controller: TabsController = new TabsController();
  private fd : resourceManager.RawFileDescriptor = {} as resourceManager.RawFileDescriptor          // 待上传的文件的文件描述符
  @State currentIndex : number = 0;
  @Provide  resultList : Array<ResultItem> = [new ResultItem("none", RecognitionType.General, 100, "xxx是...", new ImageItem('image1', {} as resourceManager.RawFileDescriptor))];

  // 将图片上传到服务器
  upload()  {
    // TODO
  }

  // 当切换Tab之后, 需要重新识别图片
  refresh() {

  }

  aboutToAppear() {
    // 根据Index传递过来的参数类型将图片上传
    // upload()
    let params = router.getParams() as Record<string, Object>;
    this.fd = params.fd as resourceManager.RawFileDescriptor;
  }

  @Builder
  TabBuilder(index: number, name: Resource) {
    Column() {
      Text(name)
        .fontColor(index == this.currentIndex ? $r('app.color.index_tab_selected_font_color') : $r('app.color.index_tab_unselected_font_color'))
        .fontSize($r('app.float.home_page_font_size'))
        .fontWeight(index == this.currentIndex ? PageConstants.FONT_WEIGHT_SELECT : PageConstants.FONT_WEIGHT_UNSELECT)
        .lineHeight(PageConstants.LINE_HEIGHT)
        .margin({
          top: PageConstants.MARGIN_TOP_TWO,
          bottom: PageConstants.MARGIN_BOTTOM
        })
      Divider()
        .strokeWidth(PageConstants.STROKE_WIDTH)
        .color($r('app.color.index_tab_selected_font_color'))
        .opacity(PageConstants.TAB_BAR_FIRST)
    }
    .width(CommonConstants.FULL_PERCENT)
    .onClick(() => {
      this.currentIndex = index;
      this.refresh()
      this.controller.changeIndex(this.currentIndex);
    })
  }

  build() {
    Column() {
      Tabs({ barPosition: BarPosition.End, controller: this.controller }) {

        // general
        TabContent() {
          ResultTabContent();
        }.tabBar(this.TabBuilder(PageConstants.TAB_BAR_FIRST, $r('app.string.index_tab_general')))

        // animal
        TabContent() {
          ResultTabContent();
        }.tabBar(this.TabBuilder(PageConstants.TAB_BAR_SECOND, $r('app.string.index_tab_animal')))

        // botany
        TabContent() {
          ResultTabContent();
        }.tabBar(this.TabBuilder(PageConstants.TAB_BAR_THIRD, $r('app.string.index_tab_botany')))

        // logo
        TabContent() {
          ResultTabContent();
        }.tabBar(this.TabBuilder(PageConstants.TAB_BAR_FOURTH, $r('app.string.index_tab_logo')))

        // money
        TabContent() {
          ResultTabContent();
        }.tabBar(this.TabBuilder(PageConstants.TAB_BAR_FIFTH, $r('app.string.index_tab_money')))

      }
      .vertical(false)
      .barMode(BarMode.Fixed)
      .barWidth(PageConstants.BAR_WIDTH)
      .barHeight(PageConstants.BAR_HEIGHT)
      .width(CommonConstants.FULL_PERCENT)
      .height(CommonConstants.FULL_PERCENT)
      .margin({ top: $r('app.float.home_tab_margin_top') })
    }
    .width(CommonConstants.FULL_PERCENT)
    .height(CommonConstants.FULL_PERCENT)
    .backgroundColor($r('app.color.index_background'))
  }
}

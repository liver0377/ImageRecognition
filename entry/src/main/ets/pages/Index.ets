import { CommonConstants } from '../common/constants/CommonConstants'
import { PageConstants } from '../common/constants/PageConstants'
import { HomeTabContent } from '../view/HomeTabContent'

@Entry
@Component
struct Index {
  private controller: TabsController = new TabsController();


  @Builder
  TabBuilder(index: number, name: Resource) {
    Column() {
      Text(name)
        .fontColor($r('app.color.index_tab_selected_font_color'))
        .fontSize($r('app.float.home_page_font_size'))
        .fontWeight(PageConstants.FONT_WEIGHT_SELECT)
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
  }

  build() {
    Column() {
      Tabs({ barPosition: BarPosition.Start, controller: this.controller }) {
        TabContent() {
          HomeTabContent();
        }.tabBar(this.TabBuilder(PageConstants.TAB_BAR_FIRST, $r('app.string.index_tab_local_video')))

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

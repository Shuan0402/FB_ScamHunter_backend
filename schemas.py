from pydantic import BaseModel, Field
from typing import List, Optional

class AdData(BaseModel):
    ad_id: str = Field(..., description="必填的廣告 ID")
    ad_context: Optional[str] = None
    ad_profile_name: Optional[str] = None
    ad_is_video: bool = False
    ad_video_preview_image: Optional[str] = None
    ad_image: Optional[str] = None
    ad_link_description: Optional[str] = None
    ad_link_url: Optional[str] = None
    ad_title: Optional[str] = None
    ad_caption: Optional[str] = None
    ad_page_id: Optional[str] = None
    ad_publisher_platform: List[str] = Field(default_factory=list)
    ad_like_count: Optional[str] = "0"
    ad_share_count: Optional[str] = "0"
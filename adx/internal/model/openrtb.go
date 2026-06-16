package model

import "errors"

type BidRequest struct {
	ID     string `json:"id"`
	Imp    []Imp  `json:"imp"`
	Site   *Site  `json:"site,omitempty"`
	Device *Device `json:"device,omitempty"`
	User   *User   `json:"user,omitempty"`
}

type Imp struct {
	ID       string  `json:"id"`
	Banner   *Banner `json:"banner,omitempty"`
	BidFloor float64 `json:"bidfloor,omitempty"`
}

type Banner struct {
	W int `json:"w,omitempty"`
	H int `json:"h,omitempty"`
}

type Site struct {
	ID     string `json:"id,omitempty"`
	Domain string `json:"domain,omitempty"`
	Page   string `json:"page,omitempty"`
}

type Device struct {
	UA  string `json:"ua,omitempty"`
	IP  string `json:"ip,omitempty"`
	Geo *Geo   `json:"geo,omitempty"`
}

type Geo struct {
	Country string `json:"country,omitempty"`
}

type User struct {
	ID string `json:"id,omitempty"`
}

type BidResponse struct {
	ID      string    `json:"id"`
	SeatBid []SeatBid `json:"seatbid,omitempty"`
	BidID   string    `json:"bidid,omitempty"`
}

type SeatBid struct {
	Bid []Bid `json:"bid,omitempty"`
}

type Bid struct {
	ID    string  `json:"id"`
	ImpID string  `json:"impid"`
	Price float64 `json:"price"`
	Adm   string  `json:"adm,omitempty"`
	AdID  string  `json:"adid,omitempty"`
	CrID  string  `json:"crid,omitempty"`
}

var ErrInvalidRequest = errors.New("invalid bid request: id is required")

func (r *BidRequest) Validate() error {
	if r.ID == "" {
		return ErrInvalidRequest
	}
	return nil
}

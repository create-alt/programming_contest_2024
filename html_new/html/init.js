Redoc.init(
{
    "openapi": "3.0.0",
    "info": {
        "title": "procon35 api",
        "description": "第35回高専プロコン競技部門で使用するAPIサーバーの仕様です。\n\n### 認証\n認証にはチームごとに事前配布したトークンを用います。\\\nトークンをクエリパラメータ(token)かヘッダー(procon-token)に指定してリクエストしてください。",
        "version": "1.0"
    },
    "paths": {
        "/problem": {
            "get": {
                "operationId": "GetProblem",
                "summary": "問題取得API",
                "description": "問題を取得するAPIです",
                "parameters": [
                    {
                        "in": "query",
                        "name": "token",
                        "description": "トークン",
                        "schema": {
                            "type": "string"
                        }
                    },
                    {
                        "in": "header",
                        "name": "Procon-Token",
                        "description": "トークン",
                        "schema": {
                            "type": "string"
                        }
                    }
                ],
                "responses": {
                    "200": {
                        "description": "正常",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "$ref": "#/components/schemas/GetProblemResponse"
                                }
                            }
                        }
                    },
                    "400": {
                        "description": "リクエストの内容が不正である場合にエラーになります"
                    },
                    "401": {
                        "description": "トークンが指定されていない、または不正である場合にエラーになります"
                    },
                    "403": {
                        "description": "競技時間外にアクセスした場合にエラーになります",
                        "content": {
                            "text/plain; charset=utf-8": {
                                "schema": {
                                    "type": "string",
                                    "enum": [
                                        "AccessTimeError"
                                    ]
                                }
                            }
                        }
                    },
                    "default": {
                        "description": "予期しないエラーです"
                    }
                }
            }
        },
        "/answer": {
            "post": {
                "operationId": "SubmitAnswer",
                "summary": "回答提出API",
                "description": "回答を提出するAPIです",
                "parameters": [
                    {
                        "in": "query",
                        "name": "token",
                        "description": "トークン",
                        "schema": {
                            "type": "string"
                        }
                    },
                    {
                        "in": "header",
                        "name": "Procon-Token",
                        "description": "トークン",
                        "schema": {
                            "type": "string"
                        }
                    }
                ],
                "requestBody": {
                    "description": "",
                    "required": true,
                    "content": {
                        "application/json": {
                            "schema": {
                                "$ref": "#/components/schemas/SubmitAnswerRequest"
                            }
                        }
                    }
                },
                "responses": {
                    "200": {
                        "description": "正常",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "$ref": "#/components/schemas/SubmitAnswerResponse"
                                }
                            }
                        }
                    },
                    "400": {
                        "description": "リクエストの内容が不正である場合にエラーになります"
                    },
                    "401": {
                        "description": "トークンが指定されていない、または不正である場合にエラーになります"
                    },
                    "403": {
                        "description": "競技時間外にアクセスした場合にエラーになります",
                        "content": {
                            "text/plain; charset=utf-8": {
                                "schema": {
                                    "type": "string",
                                    "enum": [
                                        "AccessTimeError"
                                    ]
                                }
                            }
                        }
                    },
                    "default": {
                        "description": "予期しないエラーです"
                    }
                }
            }
        }
    },
    "components": {
        "schemas": {
            "Answer": {
                "description": "回答です",
                "type": "object",
                "required": [
                    "n",
                    "ops"
                ],
                "properties": {
                    "n": {
                        "description": "手数です",
                        "type": "integer",
                        "minimum": 0
                    },
                    "ops": {
                        "type": "array",
                        "items": {
                            "$ref": "#/components/schemas/Op"
                        }
                    }
                }
            },
            "Board": {
                "description": "ボードの情報です",
                "type": "object",
                "required": [
                    "width",
                    "height",
                    "start",
                    "goal"
                ],
                "properties": {
                    "width": {
                        "description": "横のピース数です",
                        "type": "integer",
                        "minimum": 32,
                        "maximum": 256
                    },
                    "height": {
                        "description": "縦のピース数です",
                        "type": "integer",
                        "minimum": 32,
                        "maximum": 256
                    },
                    "start": {
                        "description": "ボードの初期状態です。\n長さ width の文字列が height 個の配列です",
                        "type": "array",
                        "items": {
                            "type": "string",
                            "pattern": "^[0123]+$"
                        }
                    },
                    "goal": {
                        "description": "ボードの最終状態です。\n長さ width の文字列が height 個の配列です",
                        "type": "array",
                        "items": {
                            "type": "string",
                            "pattern": "^[0123]+$"
                        }
                    }
                }
            },
            "GetProblemRequest": {
                "type": "object"
            },
            "GetProblemResponse": {
                "$ref": "#/components/schemas/Problem"
            },
            "Op": {
                "type": "object",
                "required": [
                    "p",
                    "x",
                    "y",
                    "s"
                ],
                "properties": {
                    "p": {
                        "description": "抜き型番号です",
                        "type": "integer",
                        "minimum": 0
                    },
                    "x": {
                        "description": "適用X座標です",
                        "type": "integer",
                        "minimum": -256,
                        "exclusiveMinimum": true,
                        "maximum": 256,
                        "exclusiveMaximum": true
                    },
                    "y": {
                        "description": "適用Y座標です",
                        "type": "integer",
                        "minimum": -256,
                        "exclusiveMinimum": true,
                        "maximum": 256,
                        "exclusiveMaximum": true
                    },
                    "s": {
                        "description": "寄せ方向です。\n0が上, 1が下, 2が左, 3が右です",
                        "type": "integer",
                        "enum": [
                            0,
                            1,
                            2,
                            3
                        ]
                    }
                }
            },
            "Pattern": {
                "description": "一般抜き型の情報です",
                "type": "object",
                "required": [
                    "p",
                    "width",
                    "height",
                    "cells"
                ],
                "properties": {
                    "p": {
                        "description": "抜き型番号です。\n一般抜き型は25番から始まります",
                        "type": "integer",
                        "minimum": 25
                    },
                    "width": {
                        "description": "横のセル数です",
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 256
                    },
                    "height": {
                        "description": "縦のセル数です",
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 256
                    },
                    "cells": {
                        "description": "セルの状態です。\n長さ width の文字列が height 個の配列です",
                        "type": "array",
                        "items": {
                            "type": "string",
                            "pattern": "^[01]+$"
                        }
                    }
                }
            },
            "Problem": {
                "description": "問題情報です",
                "type": "object",
                "required": [
                    "board",
                    "general"
                ],
                "properties": {
                    "startsAt": {
                        "description": "試合開始時刻です",
                        "type": "integer",
                        "minimum": 0
                    },
                    "board": {
                        "$ref": "#/components/schemas/Board"
                    },
                    "general": {
                        "description": "一般抜き型の情報です",
                        "type": "object",
                        "required": [
                            "n",
                            "patterns"
                        ],
                        "properties": {
                            "n": {
                                "description": "一般抜き型の個数です",
                                "type": "integer",
                                "minimum": 0,
                                "maximum": 256
                            },
                            "patterns": {
                                "type": "array",
                                "items": {
                                    "$ref": "#/components/schemas/Pattern"
                                }
                            }
                        }
                    }
                }
            },
            "SubmitAnswerRequest": {
                "$ref": "#/components/schemas/Answer"
            },
            "SubmitAnswerResponse": {
                "type": "object",
                "required": [
                    "revision"
                ],
                "properties": {
                    "revision": {
                        "description": "受理番号です。\nこの値が最も大きいものが最終的に受理されています",
                        "type": "integer",
                        "minimum": 0
                    }
                }
            }
        }
    }
}
)
